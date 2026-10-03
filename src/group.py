"""Group check: does a candidate record look like the other records already matched to the same S1?

A true match usually resembles the S1's other matched records (its siblings from S2/S3); a decoy is a perturbed copy
of the S1 record and often does not. For every candidate pair: similarity of the record to the S1's confident siblings
(best-S1 assignment with matcher prob > SIB_T, the record itself left out), plus rank/gap features of the scores.
A LightGBM combiner is cross-fitted on held-out train pairs (by S1 id) to measure F0.5 and pick the threshold,
then trained on all held-out pairs and applied to the test pairs.
Inputs (WORK_DIR): VAL_FILE, TEST_FILE (s1_id, q_id, prob [, x]), {train,test}_s{1,2,3}.parquet
Outputs (OUT_DIR): matching_results.tsv, group_meta.json
"""
import os, time, json
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import Levenshtein, OSA
from train import gt_pairs, decide, macro_f05, is_valid
from features import in_sample

W = os.environ.get("WORK_DIR", "work/")
O = os.environ.get("OUT_DIR", "output/")
VAL_FILE = os.environ.get("VAL_FILE", "val_preds.parquet")
TEST_FILE = os.environ.get("TEST_FILE", "test_pairs.parquet")
FRAC10 = int(os.environ.get("FRAC10", "5"))  # must match the train run
SIB_T = 0.5
NCH = 8
T0 = time.time()

def log(*a):
    print(f"[{(time.time() - T0) / 60:.1f} min]", *a, flush=True)

def text_table(split):
    cols = ["entity_id", "name_n", "addr_n"]
    d = pl.concat([pl.read_parquet(f"{W}{split}_s{i}.parquet", columns=cols) for i in (1, 2, 3)])
    return d.with_columns(pl.col("addr_n").str.extract_all(r"\b\d+\b").alias("nums"))

def sibling_feats(p, txt):
    """Per (s1_id, q_id): similarity of the record q to the S1's other confidently matched records."""
    best = p.sort("prob", descending=True).unique("q_id", keep="first")
    sib = best.filter(pl.col("prob") > SIB_T).select("s1_id", pl.col("q_id").alias("sib_id"), pl.col("prob").alias("sib_p"))
    tq = txt.rename({"entity_id": "q_id", "name_n": "nq", "addr_n": "aq", "nums": "mq"})
    ts = txt.rename({"entity_id": "sib_id", "name_n": "ns", "addr_n": "as", "nums": "ms"})
    kw = dict(workers=-1, dtype=np.float32)
    out = []
    for k in range(NCH):
        e = (p.filter(pl.col("s1_id").hash(seed=17) % NCH == k).select("s1_id", "q_id")
              .join(sib.filter(pl.col("s1_id").hash(seed=17) % NCH == k), on="s1_id")
              .filter(pl.col("q_id") != pl.col("sib_id"))
              .join(tq, on="q_id", how="left").join(ts, on="sib_id", how="left")
              .with_columns(pl.col("nq", "aq", "ns", "as").fill_null("")))
        nq, ns, aq, as_ = e["nq"].to_list(), e["ns"].to_list(), e["aq"].to_list(), e["as"].to_list()
        e = e.with_columns(
            pl.Series("s_nt", cpdist(nq, ns, scorer=fuzz.token_set_ratio, **kw)),
            pl.Series("s_nr", cpdist(nq, ns, scorer=fuzz.ratio, **kw)),
            pl.Series("s_at", cpdist(aq, as_, scorer=fuzz.token_set_ratio, **kw)),
            pl.col("mq").list.set_intersection("ms").list.len().alias("s_ni"),
            (pl.col("q_id").str.slice(0, 2) == pl.col("sib_id").str.slice(0, 2)).cast(pl.Int8).alias("s_src"))
        out.append(e.group_by("s1_id", "q_id").agg(
            pl.len().alias("g_n"), pl.col("sib_p").sum().alias("g_psum"), pl.col("s_src").sum().alias("g_same_src"),
            pl.col("s_nt").max().alias("g_nt_max"), pl.col("s_nt").mean().alias("g_nt_mean"),
            pl.col("s_nr").max().alias("g_nr_max"), pl.col("s_nr").mean().alias("g_nr_mean"),
            pl.col("s_at").max().alias("g_at_max"), pl.col("s_at").mean().alias("g_at_mean"),
            pl.col("s_ni").max().alias("g_ni_max")))
        log(f"  sibling chunk {k}: {e.height} comparisons")
    g = pl.concat(out)
    return p.join(g, on=["s1_id", "q_id"], how="left").with_columns(pl.col("g_n", "g_psum", "g_same_src").fill_null(0))

def xcols(p):
    """cross-encoder score columns: "x", or "x2", "x3", ... when several encoders are ensembled"""
    return [c for c in p.columns if c == "x" or (c[0] == "x" and c[1:].isdigit())]

# decoy signature, from the raw text so it does not depend on per-country normalisation: decoys reuse the S1 name with
# another legal form and a house number moved up (US train, same name + street: number +3 or more is a true match 15% of
# the time, -3 or more 58%); matcher features only have the absolute number difference and (France) no legal form
LEGAL_CANON = {"inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "llc": "llc", "llp": "llp",
               "lp": "lp", "pllc": "pllc", "pc": "pc", "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd",
               "pvt": "pvt", "private": "pvt", "sarl": "sarl", "sasu": "sasu", "sas": "sas", "eurl": "eurl", "sci": "sci",
               "snc": "snc", "sa": "sa", "ei": "ei"}
DEC = ["hn_sd", "hn_dir", "leg_state", "hn_shift", "nm_ext", "nm_endsub", "nm_ocr", "nm_ds", "nm_dq",
       "nm_sub1", "nm_ins1", "nm_del1", "nm_swap", "nm_biz", "nm_add_rmax", "nm_add_rmin", "nm_swap_lr"]
# filler words the noise adds to true copies are far more frequent in S2/S3 names than in S1 names (US: service 5.1x,
# center 2.5x, partners 1.9x; France: developpement, associes, international, groupe); a decoy's swapped-in word comes
# from the business vocabulary itself (martinez, crop; club, amis, ecole, union: ~0.8x). Label-free, any language.
SWAP_R = 1.3
# abbreviations that are the same word (Suzanne & Cie = Suzanne & Compagnie), and country words the noise adds as filler
# (Pornic Fetes -> Pornic SARL France, Divya (India) Wholesalers): neither is a swapped business word
WORD_CANON = {"cie": "compagnie", "ets": "etablissements", "etab": "etablissements", "ste": "sainte", "st": "saint",
              "intl": "international", "grp": "groupe", "assoc": "association", "asso": "association"}
COUNTRY_FILL = ["france", "india", "usa", "us", "america", "francaise", "french", "compagnie", "etablissements", "company"]
# street name without number / street type / stopwords: decoys keep the house number but move to another street
# (14 Rue du Jura -> 14 Allee des Conviviales); whole-address similarity dilutes this with the city
STREET_STOP = ("rue r av ave avenue allee all boulevard bd blvd bld bvd cours crs place pl impasse imp quai qu chemin ch route rte "
               "square sq passage residence res street st road rd drive dr lane ln way court ct circle cir parkway pkwy "
               "highway hwy de du des la le les l d the and of no n ndeg bis ter").split()
DEC.append("st_sim")
# house-number shifts used by the decoy generator (val errors: 64 of 400 false matches, 2 of 400 misses; +1/+2 are ambiguous)
HN_SHIFTS = [3, 4, 5, 7, 9, 11, 13, 21]
# business words decoys add to the S1 name (47 of 400 false matches, 0 misses); Services/Center/Partners are true-match noise
BIZ = ["trading", "exports", "holdings", "ventures", "group", "overseas", "agencies", "industries", "global", "media",
       "ट्रेडिंग", "ಟ್ರೇಡಿಂಗ್", "ટ્રેડિંગ", "ਟ੍ਰੇਡਿੰਗ", "ট্রেডিং", "டிரேடிங்", "ట్రేడింగ్", "ട്രേഡിംഗ്", "ଟ୍ରେଡିଂ"]
# digit/symbol read as a letter (scanning-style typo of a true match: N0vora, Pat1iputra, 8asanti)
OCR = {"0": "o", "1": "l", "8": "b", "6": "g", "5": "s", "3": "e", "4": "a", "7": "t", "|": "l"}
DOTTED = ("pllc", "sarl", "sasu", "eurl", "sas", "sci", "snc", "llc", "llp", "sa", "ei", "lp", "pc")  # longest first

def entity_sig(split):
    def load(i):
        f = f"{W}{split}_s{i}.parquet"
        cols = ["entity_id", "business_name", "business_address"] + (["country"] if "country" in pl.read_parquet_schema(f) else [])
        d = pl.read_parquet(f, columns=cols)
        return d if "country" in d.columns else d.with_columns(pl.lit("").alias("country"))
    d = pl.concat([load(i) for i in (1, 2, 3)])
    name = d["business_name"].fill_null("").str.to_lowercase().str.replace_all("à", "a")
    for w in DOTTED:  # S.A.R.L. / S A S / L.L.C. -> one token
        name = name.str.replace_all(r"\b" + r"\s?".join(c + r"\.?" for c in w) + r"(?:\b|$)", " " + w + " ")
    name = name.str.replace_all(r"[^a-z0-9]+", " ")
    toks = name.str.split(" ").list.eval(pl.element().replace_strict(LEGAL_CANON, default=None)).list.drop_nulls()
    low = d["business_name"].fill_null("").str.to_lowercase()
    # French accents are noise (Sportive/Sportîve, Fetes/Fêtes): strip them there, not for Indian scripts (vowel signs)
    low = (d.select(pl.when(pl.col("country") == "France")
                      .then(low.str.normalize("NFKD").str.replace_all(r"\p{M}", "")).otherwise(low).alias("n"))["n"])
    words = (low.str.replace_all(r"[^\w|]+", " ").str.strip_chars()
               .str.split(" ").list.eval(pl.element().filter(pl.element() != "").replace(WORD_CANON)))
    addr = d["business_address"].fill_null("").str.to_lowercase()
    addr = d.select(pl.when(pl.col("country") == "France").then(addr.str.normalize("NFKD").str.replace_all(r"\p{M}", ""))
                      .otherwise(addr).alias("a"))["a"]
    street = (addr.str.extract(r"\d{1,5}[a-z]?\b([^,]*)", 1).fill_null("").str.replace_all(r"[^a-z ]+", " ")
                  .str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() > 1) & ~pl.element().is_in(STREET_STOP)))
                  .list.join(" "))
    return pl.DataFrame({"entity_id": d["entity_id"],
                         "leg": toks.list.unique().list.sort().list.join(" "),
                         "hn": d["business_address"].str.extract(r"(?:^|\D)(\d{1,5})(?:\D|$)", 1).cast(pl.Int32, strict=False),  # 1109B -> 1109
                         "words": words, "street": street})

def word_ratio(sig):
    """per name word: frequency in S2/S3 names over frequency in S1 names (filler words the noise adds score high)"""
    w = sig.select((pl.col("entity_id").str.starts_with("S1")).alias("s1"), pl.col("words").list.unique().alias("word")).explode("word")
    n1, n23 = sig["entity_id"].str.starts_with("S1").sum(), (~sig["entity_id"].str.starts_with("S1")).sum()
    c = w.group_by("word").agg(pl.col("s1").sum().alias("c1"), (~pl.col("s1")).sum().alias("c23"))
    return c.select("word", ((pl.col("c23") / n23) / ((pl.col("c1") + 1) / n1)).log1p().cast(pl.Float32).alias("lr"), "c1")

def name_edit_feats(p, wr=None):
    """p has word lists w_s, w_q. On the words that differ between the two names (validation errors, US/India):
    nm_ext: one word is the other + 1-3 letters (Varshan/Varshana): 7.0% of wrongly accepted decoys, 0.2% of missed
    true matches; nm_endsub: same length, last 1-2 letters changed (Meenita/Meeniti): 9.0% vs 0.2%; nm_ocr: equal
    after reading digits as letters (Novora/N0vora): 0.8% vs 4.8%. nm_ds / nm_dq: number of differing words."""
    p = p.with_row_index("_r").with_columns(pl.col("w_s").list.set_difference("w_q").alias("_us"),
                                            pl.col("w_q").list.set_difference("w_s").alias("_uq"))
    x = (p.select("_r", "_us").explode("_us").join(p.select("_r", "_uq").explode("_uq"), on="_r")
          .filter(pl.col("_us").is_not_null() & pl.col("_uq").is_not_null()))
    a, b = pl.col("_us"), pl.col("_uq")
    la, lb = a.str.len_chars().cast(pl.Int32), b.str.len_chars().cast(pl.Int32)  # signed: la - lb may be negative
    oa, ob = a, b
    for k, v in OCR.items():
        oa, ob = oa.str.replace_all(k, v, literal=True), ob.str.replace_all(k, v, literal=True)
    x = x.with_columns(
        ((b.str.starts_with(a) | a.str.starts_with(b)) & (la - lb).abs().is_between(1, 3) & (pl.min_horizontal(la, lb) >= 3)).alias("e"),
        ((la == lb) & (la >= 4) & (a.str.head(la - 2) == b.str.head(lb - 2))).alias("s"),
        (oa == ob).alias("o"))
    # edit type of one-edit word pairs: decoys substitute or insert a letter (Tarain/Taraun, Nagin/Naggin), true-match
    # typos delete or swap letters (Tradng, Maritmie)
    lev = cpdist(x["_us"].to_list(), x["_uq"].to_list(), scorer=Levenshtein.distance, workers=-1) if x.height else []
    osa = cpdist(x["_us"].to_list(), x["_uq"].to_list(), scorer=OSA.distance, workers=-1) if x.height else []
    x = x.with_columns(pl.Series("_lev", lev, dtype=pl.Int32), pl.Series("_osa", osa, dtype=pl.Int32))
    lv, os_ = pl.col("_lev"), pl.col("_osa")
    x = x.with_columns(((la == lb) & (lv == 1) & ~pl.col("o")).alias("sub1"), ((lb == la + 1) & (lv == 1)).alias("ins1"),
                       ((lb == la - 1) & (lv == 1)).alias("del1"), ((os_ == 1) & (lv == 2)).alias("swap"))
    g = x.group_by("_r").agg(pl.col("e").any().alias("nm_ext"), pl.col("s").any().alias("nm_endsub"), pl.col("o").any().alias("nm_ocr"),
                             pl.col("sub1").any().alias("nm_sub1"), pl.col("ins1").any().alias("nm_ins1"),
                             pl.col("del1").any().alias("nm_del1"), pl.col("swap").any().alias("nm_swap"))
    if wr is not None:  # added (non-legal) q words: filler (high S2S3/S1 ratio) vs swapped-in business word (ratio ~1)
        legal = list(LEGAL_CANON)
        near = x.group_by("_r", "_uq").agg(pl.col("_lev").min().alias("_ml"))  # respelling of a lost S1 word?
        ad = (p.select("_r", "_uq", pl.col("_us").list.eval(pl.element().filter(~pl.element().is_in(legal))).list.len().alias("_nls"))
               .explode("_uq").filter(pl.col("_uq").is_not_null() & ~pl.col("_uq").is_in(legal))
               .join(wr, left_on="_uq", right_on="word", how="left").join(near, on=["_r", "_uq"], how="left")
               .with_columns(pl.col("lr").fill_null(0.0)))
        ga = ad.group_by("_r").agg(pl.col("lr").max().alias("nm_add_rmax"), pl.col("lr").min().alias("nm_add_rmin"),
                                   # a common business word (20+ S1 names; typos are rare) that is no filler and no respelling
                                   # (4+ letters: short acronyms like "GE" for Garderie Ecole are true-match noise)
                                   ((pl.col("lr") < np.log1p(SWAP_R)) & (pl.col("c1").fill_null(0) >= 20) & ~pl.col("_uq").is_in(COUNTRY_FILL)
                                    & (pl.col("_uq").str.len_chars() >= 4)
                                    & (pl.col("_ml").fill_null(99) > 2) & (pl.col("_nls") > 0)).any().alias("nm_swap_lr"))
        g = g.join(ga, on="_r", how="full", coalesce=True)
    return (p.join(g, on="_r", how="left")
             .with_columns(pl.col("nm_ext", "nm_endsub", "nm_ocr", "nm_sub1", "nm_ins1", "nm_del1", "nm_swap").fill_null(False).cast(pl.Int8),
                           *([pl.col("nm_add_rmax", "nm_add_rmin").fill_null(-1.0), pl.col("nm_swap_lr").fill_null(False).cast(pl.Int8)]
                             if wr is not None else []),
                           (pl.col("w_q").list.eval(pl.element().is_in(BIZ)).list.any()
                            & ~pl.col("w_s").list.eval(pl.element().is_in(BIZ)).list.any()).fill_null(False).cast(pl.Int8).alias("nm_biz"),
                           pl.col("_us").list.len().cast(pl.Int8).alias("nm_ds"), pl.col("_uq").list.len().cast(pl.Int8).alias("nm_dq"))
             .drop("_r", "_us", "_uq"))

def decoy_feats(p, split):
    if os.environ.get("DECOY_FEATS", "1") == "0":
        return p
    sig = entity_sig(split)
    wr = word_ratio(sig)
    p = (p.join(sig.rename({"entity_id": "s1_id", "leg": "l_s", "hn": "h_s", "words": "w_s", "street": "st_s"}), on="s1_id", how="left")
          .join(sig.rename({"entity_id": "q_id", "leg": "l_q", "hn": "h_q", "words": "w_q", "street": "st_q"}), on="q_id", how="left"))
    del sig
    ss, sq = p["st_s"].fill_null(""), p["st_q"].fill_null("")
    sim = cpdist(ss.to_list(), sq.to_list(), scorer=fuzz.token_set_ratio, workers=-1) if p.height else []
    empty = (pl.col("st_s").fill_null("") == "") | (pl.col("st_q").fill_null("") == "")
    p = (p.with_columns(pl.Series("st_sim", sim, dtype=pl.Float32))
          .with_columns(pl.when(empty).then(-1.0).otherwise(pl.col("st_sim")).cast(pl.Float32).alias("st_sim")).drop("st_s", "st_q"))
    d = (pl.col("h_q") - pl.col("h_s")).cast(pl.Int32)
    ls, lq = pl.col("l_s").fill_null(""), pl.col("l_q").fill_null("")
    p = p.with_columns(d.clip(-1000, 1000).alias("hn_sd"), d.sign().alias("hn_dir"), d.is_in(HN_SHIFTS).fill_null(False).cast(pl.Int8).alias("hn_shift"),
                       pl.when((ls == "") & (lq == "")).then(0).when((ls == "") | (lq == "")).then(1)
                         .when(ls == lq).then(2).otherwise(3).cast(pl.Int8).alias("leg_state"),
                       pl.col("w_s", "w_q").fill_null([])).drop("l_s", "l_q", "h_s", "h_q")
    return name_edit_feats(p, wr).drop("w_s", "w_q")

def score_feats(p):
    xs = xcols(p)
    sc = ["prob"] + xs + [c for c in ("p1",) if c in p.columns]
    if xs:
        p = p.with_columns((pl.col("prob") + pl.mean_horizontal(xs)).alias("px"))
        sc.append("px")
    f = []
    for c in sc:
        f += [pl.col(c).rank("ordinal", descending=True).over("q_id").alias(f"rk_{c}"),
              (pl.col(c).max().over("q_id") - pl.col(c)).alias(f"gapq_{c}"),
              (pl.col(c) - pl.col(c).sort(descending=True).slice(1, 1).first().over("q_id").fill_null(0)).alias(f"gap2_{c}"),
              pl.col(c).rank("ordinal", descending=True).over("s1_id").alias(f"rks_{c}"),
              (pl.col(c).max().over("s1_id") - pl.col(c)).alias(f"gaps_{c}"),
              (pl.col(c) > SIB_T).sum().over("s1_id").alias(f"ns_{c}")]
    f += [pl.len().over("q_id").alias("nq"), pl.len().over("s1_id").alias("ns"),
          pl.col("q_id").str.starts_with("S2").cast(pl.Int8).alias("is_s2")]
    return p.with_columns(f)

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())

def fit_eval(va, CF, y, fold):
    X = va.select(CF).to_numpy().astype(np.float32)
    oof = np.zeros(va.height, dtype=np.float32)
    rounds = []
    for k in range(4):
        trn, tst = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[trn], y[trn], feature_name=CF), 3000,
                      valid_sets=[lgb.Dataset(X[tst], y[tst], feature_name=CF)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[tst] = m.predict(X[tst]); rounds.append(m.best_iteration)
    comb = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=CF), int(np.mean(rounds) * 1.1))
    return oof, comb

def best_t(va, col, vtruth, ids):
    ts = np.round(np.arange(0.3, 0.96, 0.025), 3)
    p = va.select("s1_id", "q_id").with_columns(pl.Series("prob", col))
    scs = [macro_f05(decide(p, t), vtruth, ids) for t in ts]
    return float(ts[int(np.argmax(scs))]), max(scs)

def round2(p, p1, txt):
    """Second pass: siblings re-picked with the round-1 combined score p1 (kept as a feature next to prob)."""
    base = p.select(["s1_id", "q_id", "prob"] + xcols(p) + [c for c in DEC if c in p.columns]).with_columns(pl.Series("p1", p1))
    q = sibling_feats(base.rename({"prob": "p0"}).rename({"p1": "prob"}), txt)  # sibling_feats picks siblings by "prob"
    return score_feats(q.rename({"prob": "p1"}).rename({"p0": "prob"}))

SEEN = ("US", "India")
# French legal forms as written in the raw names (S.A.R.L., SÀRL, [SAS], EURL, ...)
FR_LEGAL_RAW = (r"(?:^|[^\w])(s\.?\s?a\.?\s?r\.?\s?l|s\.?\s?a\.?\s?s\.?\s?u|s\.?\s?a\.?\s?s|e\.?\s?u\.?\s?r\.?\s?l"
                r"|s\.?\s?c\.?\s?i|s\.?\s?n\.?\s?c|s\.?\s?a|e\.?\s?i)\.?(?:$|[^\w])")

def drop_decoys(te):
    """Unseen countries: zero the score of pairs whose legal forms and house numbers both differ. prep.py strips French
    legal forms, so the model only sees the number there; in US train, same-name pairs with a different legal form AND
    a different house number are true 0.1% of the time (0.87 when the number agrees)."""
    raw = pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id", "country", "business_name", "business_address"])
                     for i in (1, 2, 3)]).filter(~pl.col("country").is_in(SEEN))
    leg = (pl.col("business_name").str.to_lowercase().str.replace_all("à", "a").str.extract(FR_LEGAL_RAW, 1)
             .str.replace_all(r"[^a-z]", ""))
    raw = raw.select("entity_id", leg.alias("leg"), pl.col("business_address").str.extract(r"(\d+)").alias("hn"))
    un = te.filter(~pl.col("country").is_in(SEEN))
    un = (un.join(raw.rename({"entity_id": "s1_id", "leg": "l1", "hn": "h1"}), on="s1_id", how="left")
            .join(raw.rename({"entity_id": "q_id", "leg": "lq", "hn": "hq"}), on="q_id", how="left"))
    decoy = (pl.col("l1") != pl.col("lq")) & (pl.col("h1") != pl.col("hq"))  # null (missing) never counts as different
    un = un.with_columns(pl.when(decoy.fill_null(False)).then(0.0).otherwise(pl.col("prob")).alias("prob"))
    log(f"decoys zeroed (unseen countries): {un.filter(pl.col('prob') == 0).height - te.filter(~pl.col('country').is_in(SEEN) & (pl.col('prob') == 0)).height}")
    return pl.concat([te.filter(pl.col("country").is_in(SEEN)), un.select(te.columns)])

def calibrated(te, s1t, t):
    """decide() with its own threshold for each country unseen in training (France): the threshold that gives it the
    same matches per S1 as the trained countries get at t. Train truth has identical match rates for US and India,
    while France with the shared t got ~0.3 extra matches per S1 (false matches, as it has fewer S2/S3 per S1)."""
    te = te.join(s1t.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    if os.environ.get("FR_DECOY", "1") != "0":
        te = drop_decoys(te)
    best = te.sort("prob", descending=True).unique("q_id", keep="first")
    n1 = s1t.group_by("country").len()
    n = dict(zip(n1["country"].to_list(), n1["len"].to_list()))
    seen = best.filter(pl.col("country").is_in(SEEN))
    rate = (seen["prob"] > t).sum() / sum(n.get(c, 0) for c in SEEN)
    tmap = {"_all": t}
    for c in sorted(set(n) - set(SEEN)):
        ps = np.sort(best.filter(pl.col("country") == c)["prob"].to_numpy())[::-1]
        k = int(round(rate * n[c]))
        tmap[c] = float(max(t, ps[k])) if k < len(ps) else t
        log(f"calibrate {c}: {n[c]} S1, {(ps > t).sum() / n[c]:.3f} matches/S1 at t {t} -> t {tmap[c]:.4f} "
            f"({(ps > tmap[c]).sum() / n[c]:.3f}/S1, trained countries {rate:.3f})")
    thr = pl.col("country").replace_strict(tmap, default=t, return_dtype=pl.Float64)
    return best.filter(pl.col("prob") > thr).select("s1_id", "q_id")

def one_owner(p):
    """A record belongs to at most one S1: turn its candidates' probs into shares of one owner, with "no match" as an
    option, q_i = o_i / (1 + sum_j o_j) with odds o = p / (1 - p). Two strong claimants then each get less."""
    o = (pl.col("prob").clip(1e-6, 1 - 1e-6) / (1 - pl.col("prob").clip(1e-6, 1 - 1e-6)))
    return p.with_columns((o / (1 + o.sum().over("q_id"))).alias("prob"))

def expf_decide(p, floor=0.02):
    """Macro F0.5 is averaged per S1, so the best cut depends on each S1's whole candidate list, not on one global
    threshold. Each record first goes to its best S1 (as in decide()); then per S1, with its candidates sorted by prob,
    take the top k that maximises the expected F0.5 ~ 1.25 * S_k / (k + 0.25 * N) (S_k: sum of the top-k probs, N: sum
    of all its probs = expected number of true matches), or none when P(no true match) = prod(1 - p) is higher."""
    b = (p.sort("prob", descending=True).unique("q_id", keep="first").filter(pl.col("prob") > floor)
          .sort(["s1_id", "prob"], descending=[False, True]))
    b = b.with_columns(pl.int_range(1, pl.len() + 1).over("s1_id").alias("_k"), pl.col("prob").cum_sum().over("s1_id").alias("_S"),
                       pl.col("prob").sum().over("s1_id").alias("_N"),
                       (1 - pl.col("prob")).clip(1e-9, 1).log().sum().over("s1_id").exp().alias("_P0"))
    b = b.with_columns((1.25 * pl.col("_S") / (pl.col("_k") + 0.25 * pl.col("_N"))).alias("_ef"))
    b = b.with_columns(pl.col("_ef").max().over("s1_id").alias("_efmax"))
    b = b.with_columns(pl.when(pl.col("_ef") == pl.col("_efmax")).then(pl.col("_k")).min().over("s1_id").alias("_kb"))
    return b.filter((pl.col("_k") <= pl.col("_kb")) & (pl.col("_efmax") > pl.col("_P0"))).select("s1_id", "q_id")

def main():
    os.makedirs(O, exist_ok=True)
    truth = gt_pairs()
    va0 = pl.read_parquet(W + VAL_FILE).select(pl.exclude("country"))
    log("val pairs", va0.height, "cols", va0.columns)
    txt = text_table("train")
    va = decoy_feats(score_feats(sibling_feats(va0, txt)), "train").join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "q_id"], how="left")                                             .with_columns(pl.col("y").fill_null(0))
    CF = [c for c in va.columns if c not in ("s1_id", "q_id", "y")]
    s1 = pl.read_parquet(W + "train_s1.parquet", columns=["entity_id", "country"]).filter(
        is_valid(pl.col("entity_id")) & in_sample(pl.col("entity_id"), FRAC10))
    ids = s1["entity_id"]
    vtruth = truth.filter(pl.col("s1_id").is_in(ids.implode()))
    fold = (va["s1_id"].hash(seed=13) % 4).to_numpy()
    y = va["y"].to_numpy()
    oof1, comb1 = fit_eval(va, CF, y, fold)
    res = {"matcher only": best_t(va, va["prob"].to_numpy(), vtruth, ids)}
    for c in xcols(va):
        res[f"cross-encoder {c} only"] = best_t(va, va[c].to_numpy(), vtruth, ids)
    res["group round 1"] = best_t(va, oof1, vtruth, ids)
    # round 2: siblings from the round-1 score (out-of-fold on val, so no label leak)
    v2 = round2(va.select(va0.columns + [c for c in DEC if c in va.columns]), oof1, txt).join(va.select("s1_id", "q_id", "y"), on=["s1_id", "q_id"])
    del txt
    CF2 = [c for c in v2.columns if c not in ("s1_id", "q_id", "y")]
    fold2, y2 = (v2["s1_id"].hash(seed=13) % 4).to_numpy(), v2["y"].to_numpy()
    oof2, comb2 = fit_eval(v2, CF2, y2, fold2)
    res["group round 2"] = best_t(v2, oof2, vtruth, ids)
    for k, v in res.items():
        log(f"{k}: best t {v[0]} F0.5 {v[1]:.4f}")
    use2 = res["group round 2"][1] > res["group round 1"][1] or os.environ.get("GROUP_FORCE2") is not None
    t_best = res["group round 2" if use2 else "group round 1"][0]
    oo, vv = (oof2, v2) if use2 else (oof1, va)
    for c in s1["country"].unique().sort().to_list():
        cid = s1.filter(pl.col("country") == c)["entity_id"]
        p = vv.select("s1_id", "q_id").with_columns(pl.Series("prob", oo)).filter(pl.col("s1_id").is_in(cid.implode()))
        log("  ", c, round(macro_f05(decide(p, t_best), vtruth.filter(pl.col("s1_id").is_in(cid.implode())), cid), 4))
    # per-S1 expected-F0.5 decision instead of the single threshold, kept only if it scores higher on held-out val
    pv = vv.select("s1_id", "q_id").with_columns(pl.Series("prob", oo))
    f_thr = res["group round 2" if use2 else "group round 1"][1]
    f_raw = macro_f05(expf_decide(pv), vtruth, ids)
    f_own = macro_f05(expf_decide(one_owner(pv)), vtruth, ids)
    f_exp, use_own = max((f_raw, False), (f_own, True))
    use_exp = f_exp > f_thr and os.environ.get("EXPF", "1") != "0"
    res["expected-F decision"] = (None, f_raw)
    res["expected-F one-owner"] = (None, f_own)
    log(f"expected-F: raw {f_raw:.4f}, one-owner {f_own:.4f} vs threshold {f_thr:.4f} -> "
        f"{('USE ' + ('one-owner' if use_own else 'raw')) if use_exp else 'keep threshold'}")
    for c in s1["country"].unique().sort().to_list():
        cid = s1.filter(pl.col("country") == c)["entity_id"]
        pc = pv.filter(pl.col("s1_id").is_in(cid.implode()))
        log("   expected-F", c, round(macro_f05(expf_decide(one_owner(pc) if use_own else pc),
                                                 vtruth.filter(pl.col("s1_id").is_in(cid.implode())), cid), 4))
    del pv
    comb = comb2 if use2 else comb1
    imp = sorted(zip(comb.feature_name(), comb.feature_importance("gain")), key=lambda z: -z[1])
    log("using round", 2 if use2 else 1, "importance", [(a, int(b)) for a, b in imp[:15]])
    json.dump({"threshold": t_best, "scores": res, "round": 2 if use2 else 1, "features": CF2 if use2 else CF},
              open(O + "group_meta.json", "w"))
    del va, v2

    te0 = pl.read_parquet(W + TEST_FILE)
    te0 = te0.select(["s1_id", "q_id", "prob"] + xcols(te0))
    txt = text_table("test")
    te = decoy_feats(score_feats(sibling_feats(te0, txt)), "test")
    pt = comb1.predict(te.select(CF).to_numpy().astype(np.float32))
    if use2:
        te = round2(te.select(te0.columns + [c for c in DEC if c in te.columns]), pt, txt)
        pt = comb2.predict(te.select(CF2).to_numpy().astype(np.float32))
    del txt
    te = te.select("s1_id", "q_id").with_columns(pl.Series("prob", pt))
    te.write_parquet(O + "test_scores.parquet")
    s1t = pl.read_parquet(W + "test_s1.parquet", columns=["entity_id", "country"])
    pred = calibrated(te, s1t, t_best) if os.environ.get("FR_CALIB", "1") != "0" else decide(te, t_best)
    if use_exp:  # trained countries: expected-F decision; unseen (France) keeps its calibrated threshold
        seen_ids = s1t.filter(pl.col("country").is_in(SEEN)).select(pl.col("entity_id").alias("s1_id"))
        ts = te.join(seen_ids, on="s1_id")
        pred = pl.concat([pred.join(seen_ids, on="s1_id", how="anti"), expf_decide(one_owner(ts) if use_own else ts)])
        log("expected-F decision applied to", ", ".join(SEEN))
    g = pred.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("matched_entity_ids"))
    (s1t.join(g, left_on="entity_id", right_on="s1_id", how="left").sort("entity_id")
        .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
        .write_csv(O + "matching_results.tsv", separator="	", quote_style="never"))
    log("test pairs", te.height, "matches", pred.height, "s1 with match", pred["s1_id"].n_unique(), "/", s1t.height)

if __name__ == "__main__":
    main()
