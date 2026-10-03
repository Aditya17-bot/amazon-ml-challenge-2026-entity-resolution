"""Local post-processing of finished group-run outputs (no retraining).

  python variants.py nocalib <group_out_dir> <dst_dir>
      France re-decided at the shared validation threshold (decoy rule kept) instead of the calibrated France threshold;
      US/India rows unchanged.
  python variants.py merge <us_india_out_dir> <france_out_dir> <dst_dir>
      US/India rows (matches and candidates) from one run, France rows from another. US/India text is identical in
      both pipelines (only French legal forms changed), so each country keeps an internally consistent run.
Needs WORK_DIR with raw test_s{1,2,3}.parquet (entity_id, country, business_name, business_address).
"""
import json, os, sys
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # group.py / train.py sit next to this file
ARGS, sys.argv = sys.argv, sys.argv[:1]  # train.py parses sys.argv at import
import group as g  # noqa: E402
from train import decide  # noqa: E402
sys.argv = ARGS

W = os.environ.get("WORK_DIR", "../work/frtest/")
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)

def countries():
    return pl.read_parquet(W + "test_s1.parquet", columns=["entity_id", "country"])

def write(match, cand, dst):
    os.makedirs(dst, exist_ok=True)
    match.sort("source1_entity_id").write_csv(dst + "/matching_results.tsv", separator="\t", quote_style="never")
    cand.sort("source1_entity_id").write_csv(dst + "/candidate_pairs.tsv", separator="\t", quote_style="never")

def nocalib(src, dst):
    t = json.load(open(src + "/group_meta.json"))["threshold"]
    s1t = countries()
    te = pl.read_parquet(src + "/test_scores.parquet").join(s1t.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    fr = g.drop_decoys(te).filter(~pl.col("country").is_in(g.SEEN))
    pf = decide(fr.select("s1_id", "q_id", "prob"), t)
    gf = pf.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    frids = s1t.filter(~pl.col("country").is_in(g.SEEN))["entity_id"]
    m = rd(src + "/matching_results.tsv")
    m_fr = (pl.DataFrame({"source1_entity_id": frids}).join(gf, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    out = pl.concat([m.filter(~pl.col("source1_entity_id").is_in(frids.implode())), m_fr])
    print(f"France threshold {t} (no calibration): {pf.height} France matches, {pf.height / len(frids):.3f}/S1")
    write(out, rd(src + "/candidate_pairs.tsv"), dst)

def frtarget(src, dst, rate):
    """France re-decided (decoy rule kept) at the threshold that gives `rate` matches per France S1."""
    import numpy as np
    s1t = countries()
    te = pl.read_parquet(src + "/test_scores.parquet").join(s1t.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    fr = g.drop_decoys(te).filter(~pl.col("country").is_in(g.SEEN))
    best = fr.sort("prob", descending=True).unique("q_id", keep="first")
    frids = s1t.filter(~pl.col("country").is_in(g.SEEN))["entity_id"]
    ps = np.sort(best["prob"].to_numpy())[::-1]
    t = float(ps[int(round(float(rate) * len(frids)))])
    pf = best.filter(pl.col("prob") > t).select("s1_id", "q_id")
    gf = pf.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m = rd(src + "/matching_results.tsv")
    m_fr = (pl.DataFrame({"source1_entity_id": frids}).join(gf, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    print(f"France target {rate}/S1 -> threshold {t:.4f}, {pf.height} matches")
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(frids.implode())), m_fr]), rd(src + "/candidate_pairs.tsv"), dst)

def frswap(src, dst):
    """France: drop pairs with the word-swap decoy flag (nm_swap_lr), then the usual decoy rule + calibration."""
    t = json.load(open(src + "/group_meta.json"))["threshold"]
    s1t = countries()
    te = pl.read_parquet(src + "/test_scores.parquet").join(s1t.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    cand = te.filter(~pl.col("country").is_in(g.SEEN) & (pl.col("prob") > 0.3)).select("s1_id", "q_id", "prob")
    f = g.decoy_feats(cand, "test")
    # swap flag only: the street flag misfires on multi-number French addresses (appt 11 - 1er etage ...)
    f = f.filter(pl.col("nm_swap_lr") == 1).select("s1_id", "q_id")
    print(f"France pairs flagged as word-swap decoys (prob > 0.3): {f.height}")
    te = (te.join(f.with_columns(pl.lit(1).alias("_f")), on=["s1_id", "q_id"], how="left")
            .with_columns(pl.when(pl.col("_f") == 1).then(0.0).otherwise(pl.col("prob")).alias("prob")).drop("_f", "country"))
    pred = g.calibrated(te, s1t, t)
    frids = s1t.filter(~pl.col("country").is_in(g.SEEN))["entity_id"]
    pf = pred.filter(pl.col("s1_id").is_in(frids.implode()))
    gf = pf.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m = rd(src + "/matching_results.tsv")
    m_fr = (pl.DataFrame({"source1_entity_id": frids}).join(gf, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    print(f"France: {pf.height} matches, {pf.height / len(frids):.3f}/S1")
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(frids.implode())), m_fr]), rd(src + "/candidate_pairs.tsv"), dst)

def fraggr(src, dst, t_fr="0.9", biz="1", fillx="0"):
    """France, all decoy signatures at once: word swap; house number differs (both addresses have exactly one number:
    France noise almost never changes numbers, 1.7% of accepted pairs vs 10-11% in US/India); same number but another
    street; then France's own stricter threshold t_fr (its borderline pairs measured ~57% correct on the leaderboard)."""
    t_fr = float(t_fr)
    s1t = countries()
    te = pl.read_parquet(src + "/test_scores.parquet").join(s1t.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    cand = te.filter(~pl.col("country").is_in(g.SEEN) & (pl.col("prob") > 0.3)).select("s1_id", "q_id", "prob")
    f = g.decoy_feats(cand, "test")
    raw = pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id", "business_address"]) for i in (1, 2, 3)])
    nn = raw.select("entity_id", pl.col("business_address").fill_null("").str.extract_all(r"(?:^|\D)(\d{1,5})(?:\D|$)").list.len().alias("nn"))
    f = (f.join(nn.rename({"entity_id": "s1_id", "nn": "n_s"}), on="s1_id", how="left")
          .join(nn.rename({"entity_id": "q_id", "nn": "n_q"}), on="q_id", how="left"))
    single = (pl.col("n_s") == 1) & (pl.col("n_q") == 1)
    swap = pl.col("nm_swap_lr") == 1
    numdiff = single & pl.col("hn_sd").is_not_null() & (pl.col("hn_sd") != 0)
    street = single & (pl.col("hn_sd") == 0) & (pl.col("st_sim") >= 0) & (pl.col("st_sim") < 50)
    f = f.with_columns(swap.alias("swap"), numdiff.fill_null(False).alias("numdiff"), street.fill_null(False).alias("street"))
    # French counterparts of the business words decoys add in US/India (Group: 47 false matches, 0 misses): "groupe"
    # added; "fils" (& Fils, ratio 1.46 like group/exports, not filler 1.8-5.1) when it replaces an S1 word
    nm = pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id", "business_name"]) for i in (1, 2, 3)])
    wd = nm.select("entity_id", pl.col("business_name").fill_null("").str.to_lowercase().str.normalize("NFKD")
                   .str.replace_all(r"\p{M}", "").str.replace_all(r"[^\w]+", " ").str.strip_chars().str.split(" ").alias("w"))
    f = (f.join(wd.rename({"entity_id": "s1_id", "w": "ws"}), on="s1_id", how="left").join(wd.rename({"entity_id": "q_id", "w": "wq"}), on="q_id", how="left"))
    legal = list(g.LEGAL_CANON) + ["et", "fils", "groupe", "de", "du", "des", "la", "le", "les"]
    lost = pl.col("ws").list.set_difference("wq").list.eval(pl.element().filter(~pl.element().is_in(legal))).list.len() > 0
    adds = lambda w: pl.col("wq").list.contains(w) & ~pl.col("ws").list.contains(w)
    f = f.with_columns((adds("groupe") | (adds("fils") & lost)).fill_null(False).alias("biz_fr")).drop("ws", "wq")
    print("France flags (prob > 0.3): swap", f["swap"].sum(), "| number differs", f["numdiff"].sum(), "| street changed", f["street"].sum(),
          "| groupe / fils", f["biz_fr"].sum())
    fl = f.filter(pl.col("swap") | pl.col("numdiff") | pl.col("street") | (pl.col("biz_fr") & (biz == "1"))).select("s1_id", "q_id")
    te = (te.join(fl.with_columns(pl.lit(1).alias("_f")), on=["s1_id", "q_id"], how="left")
            .with_columns(pl.when(pl.col("_f") == 1).then(0.0).otherwise(pl.col("prob")).alias("prob")).drop("_f"))
    fr = g.drop_decoys(te).filter(~pl.col("country").is_in(g.SEEN))
    # acronym / domain copies (EHPAD Jeanne -> EJ, csainte.com) are true-match noise: they keep the shared threshold
    t_all = json.load(open(src + "/group_meta.json"))["threshold"]
    nq = nm.select(pl.col("entity_id").alias("q_id"), pl.col("business_name").fill_null("").str.to_lowercase().alias("_nq"))
    best = fr.select("s1_id", "q_id", "prob").sort("prob", descending=True).unique("q_id", keep="first").join(nq, on="q_id", how="left")
    short = (pl.col("_nq").str.replace_all(r"[^a-z]", "").str.len_chars().is_between(2, 4) & ~pl.col("_nq").str.contains(" ")) | pl.col("_nq").str.contains(r"\.com|www\.|\.fr\b")
    if fillx == "1":
        # French filler additions are true-match noise the US/India model does not know (leaderboard: dropping
        # Developpement/Associes pairs cost 0.0005, keeping Groupe/Fils pairs gained 0.0007): shared threshold for them too
        best = (best.join(wd.rename({"entity_id": "s1_id", "w": "_ws"}), on="s1_id", how="left")
                    .join(wd.rename({"entity_id": "q_id", "w": "_wq"}), on="q_id", how="left"))
        added = pl.lit(False)
        for w in ("developpement", "associes", "groupe", "fils", "services", "international", "distribution"):
            added = added | (pl.col("_wq").list.contains(w) & ~pl.col("_ws").list.contains(w))
        short = short | added.fill_null(False)
    pf = best.filter(pl.col("prob") > pl.when(short).then(t_all).otherwise(t_fr)).select("s1_id", "q_id")
    frids = s1t.filter(~pl.col("country").is_in(g.SEEN))["entity_id"]
    gf = pf.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m = rd(src + "/matching_results.tsv")
    m_fr = (pl.DataFrame({"source1_entity_id": frids}).join(gf, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    print(f"France threshold {t_fr}: {pf.height} matches, {pf.height / len(frids):.3f}/S1, empty S1 {1 - pf['s1_id'].n_unique() / len(frids):.3f}")
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(frids.implode())), m_fr]), rd(src + "/candidate_pairs.tsv"), dst)

def ustrim(scores_dir, src, dst, excess="0.048", country="US"):
    """US: same matcher, same threshold gives 3.358 matches/S1 on test vs 3.310 on validation (India 3.308 vs 3.301),
    although US test has fewer S2/S3 records per S1 than India: the excess is decoys from test's denser distractors.
    Remove the `excess` x (US S1 count) least confident US matches, only from S1s that keep another match."""
    s1t = countries()
    us = s1t.filter(pl.col("country") == country)["entity_id"]
    m = rd(src + "/matching_results.tsv")
    p = (m.filter(pl.col("source1_entity_id").is_in(us.implode()) & (pl.col("matched_entity_ids").fill_null("") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"})
          .join(pl.read_parquet(scores_dir + "/test_scores.parquet"), on=["s1_id", "q_id"], how="left")
          .with_columns(pl.len().over("s1_id").alias("n"), pl.col("prob").rank("ordinal").over("s1_id").alias("rk")))
    k = int(round(float(excess) * len(us)))
    drop = p.filter((pl.col("n") >= 2) & (pl.col("rk") < pl.col("n"))).sort("prob").head(k).select("s1_id", "q_id")
    keep = p.join(drop, on=["s1_id", "q_id"], how="anti")
    gu = keep.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m_us = (pl.DataFrame({"source1_entity_id": us}).join(gu, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    print(f"{country}: dropped {drop.height} least-confident matches (S1 keeps another); {p.height} -> {keep.height}, {keep.height / len(us):.3f}/S1")
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(us.implode())), m_us]), rd(src + "/candidate_pairs.tsv"), dst)

def dropwords(src, dst, words):
    """France: drop pairs whose record adds one of `words` (comma list, accent-free) that the S1 name lacks."""
    ws = words.split(",")
    s1t = countries()
    fr = s1t.filter(~pl.col("country").is_in(g.SEEN))["entity_id"]
    m = rd(src + "/matching_results.tsv")
    p = (m.filter(pl.col("source1_entity_id").is_in(fr.implode()) & (pl.col("matched_entity_ids").fill_null("") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))
    nm = pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id", "business_name"]) for i in (1, 2, 3)])
    wd = nm.select("entity_id", pl.col("business_name").fill_null("").str.to_lowercase().str.normalize("NFKD")
                   .str.replace_all(r"\p{M}", "").str.replace_all(r"[^\w]+", " ").str.strip_chars().str.split(" ").alias("w"))
    p = p.join(wd.rename({"entity_id": "s1_id", "w": "ws"}), on="s1_id", how="left").join(wd.rename({"entity_id": "q_id", "w": "wq"}), on="q_id", how="left")
    hit = pl.lit(False)
    for w in ws:
        hit = hit | (pl.col("wq").list.contains(w) & ~pl.col("ws").list.contains(w))
    keep = p.filter(~hit.fill_null(False))
    print(f"France: dropped {p.height - keep.height} pairs adding {ws}")
    gf = keep.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m_fr = (pl.DataFrame({"source1_entity_id": fr}).join(gf, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(fr.implode())), m_fr]), rd(src + "/candidate_pairs.tsv"), dst)

def uiswap(scores_dir, src, dst, pmax="0.99"):
    """US/India: drop accepted pairs (prob < pmax) carrying the word-swap decoy flag (test has ~2x the decoys of val)."""
    s1t = countries()
    ui = s1t.filter(pl.col("country").is_in(g.SEEN))["entity_id"]
    m = rd(src + "/matching_results.tsv")
    p = (m.filter(pl.col("source1_entity_id").is_in(ui.implode()) & (pl.col("matched_entity_ids").fill_null("") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"})
          .join(pl.read_parquet(scores_dir + "/test_scores.parquet"), on=["s1_id", "q_id"], how="left"))
    c = p.filter(pl.col("prob") < float(pmax)).select("s1_id", "q_id", "prob")
    f = g.decoy_feats(c, "test").filter(pl.col("nm_swap_lr") == 1).select("s1_id", "q_id")
    keep = p.join(f, on=["s1_id", "q_id"], how="anti")
    print(f"US/India: dropped {f.height} swap-flagged pairs (prob < {pmax})")
    gu = keep.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    m_ui = (pl.DataFrame({"source1_entity_id": ui}).join(gu, left_on="source1_entity_id", right_on="s1_id", how="left")
              .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    write(pl.concat([m.filter(~pl.col("source1_entity_id").is_in(ui.implode())), m_ui]), rd(src + "/candidate_pairs.tsv"), dst)

def merge(src_ui, src_fr, dst):
    frids = countries().filter(~pl.col("country").is_in(g.SEEN))["entity_id"].implode()
    pick = lambda f: pl.concat([rd(src_ui + "/" + f).filter(~pl.col("source1_entity_id").is_in(frids)),
                                rd(src_fr + "/" + f).filter(pl.col("source1_entity_id").is_in(frids))])
    write(pick("matching_results.tsv"), pick("candidate_pairs.tsv"), dst)
    print("merged: US/India from", src_ui, "| France from", src_fr)

if __name__ == "__main__":
    {"nocalib": nocalib, "merge": merge, "frtarget": frtarget, "frswap": frswap, "fraggr": fraggr, "ustrim": ustrim, "dropwords": dropwords, "uiswap": uiswap}[sys.argv[1]](*sys.argv[2:])
