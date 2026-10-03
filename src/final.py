"""Final post-processing that turns the two group runs into the submitted file (leaderboard 0.984261).
No retraining: only the scored test pairs (test_scores.parquet) and the raw test text are used.

  python final.py <us_india_group_out> <france_group_out> <dst_dir>

  us_india_group_out: group.py output of the run with all five cross-encoders (x2..x6)
  france_group_out:   group.py output of the France rebuild (encoders x5, x6; French legal forms kept)

Steps (each one was tested on the leaderboard, see Documentation_template.md, section 5):
  1. France decoy rules at threshold 0.97 (variants.fraggr), merged with US/India (variants.merge)
  2. brand copies: an S2/S3 record whose name is one made-up word sharing nothing with the S1 name, at the same
     house number and address, rejected by the threshold -> accepted when prob > 0.5 (US/India)
  3. France: decoy rules without the Groupe/Fils flag, French filler words keep the shared threshold
     (fraggr t_fr=0.97 biz=0 fillx=1), plus France brand copies with prob > 0.5
  4. France: records adding a filler word at the same house number and address (sim >= 85), prob > 0.3
  5. France: pairs a 0.725 threshold would accept, not flagged by any decoy rule, whose record has no competing
     S1 above 0.3
Every added pair belongs to a still-unmatched record, so each S2/S3 record keeps at most one S1.
Needs WORK_DIR with raw test_s{1,2,3}.parquet (as variants.py).
"""
import os, shutil, sys
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import variants as v
from variants import W, rd

FILL = ["developpement", "associes", "groupe", "fils", "services", "international", "distribution"]
FR_AREA = r"\b(pays de la loire|nouvelle aquitaine|hauts de france|loire atlantique|gironde|nord|pas de calais)\b"
STREET = r"\b(rue|r|avenue|av|ave|bd|boulevard|allee|all|place|pl|route|rte|no|n|de|du|des|la|le|les|l|d)\b"
nrm = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}", "")

def pairs(m):
    return (m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))

def raw():
    return pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (1, 2, 3)])

def france_ids():
    return v.countries().filter(pl.col("country") == "France")["entity_id"].implode()

def add_pairs(src, add, dst, label):
    """src + add, where add keeps only records not matched anywhere in src (one owner per record)."""
    m = rd(src + "/matching_results.tsv")
    p = pairs(m)
    add = add.select("s1_id", "q_id").join(p.select("q_id"), on="q_id", how="anti").unique("q_id", keep="first")
    print(f"{label}: {add.height} pairs added")
    p = pl.concat([p, add])
    assert p["q_id"].n_unique() == p.height, "record matched twice"
    gr = p.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("m"))
    out = (m.select("source1_entity_id").join(gr, left_on="source1_entity_id", right_on="s1_id", how="left")
            .select("source1_entity_id", pl.col("m").fill_null("").alias("matched_entity_ids")))
    v.write(out, rd(src + "/candidate_pairs.tsv"), dst)

def take_france(src_ui, src_fr, dst):
    """US/India rows from src_ui, France rows from src_fr (both candidate sets are the same file here)."""
    fr = france_ids()
    m = pl.concat([rd(src_ui + "/matching_results.tsv").filter(~pl.col("source1_entity_id").is_in(fr)),
                   rd(src_fr + "/matching_results.tsv").filter(pl.col("source1_entity_id").is_in(fr))])
    v.write(m, rd(src_ui + "/candidate_pairs.tsv"), dst)

def brand_rejected(ui_scores, fr_scores, final_dir):
    """Per country: records whose best S1 was not accepted, renamed to a single made-up word (no shared word, name
    similarity < 50) at the same house number and address (token-set similarity >= 90)."""
    s1 = v.countries()
    leg = r"\b(inc|llc|corp|corporation|co|company|ltd|limited|pvt|private|lp|llp|pllc|pc|sarl|sas|sasu|eurl|sci|snc|sa|ei)\b"
    info = raw().select("entity_id",
        nrm("business_name").str.replace_all(r"[^\w]+", " ").str.replace_all(leg, " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 1)).alias("w"),
        nrm("business_address").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().alias("a"),
        pl.col("business_address").str.extract(r"(?:^|\D)(\d{1,5})(?:\D|$)", 1).alias("hn"))
    fin = pairs(rd(final_dir + "/matching_results.tsv"))
    out = []
    for c, sc in (("US", ui_scores), ("India", ui_scores), ("France", fr_scores)):
        te = pl.read_parquet(sc + "/test_scores.parquet").join(s1.rename({"entity_id": "s1_id"}), on="s1_id").filter(pl.col("country") == c)
        best = te.sort("prob", descending=True).unique("q_id", keep="first")
        rej = best.join(fin, on=["s1_id", "q_id"], how="anti").filter(pl.col("prob") > 0.05)
        rej = (rej.join(info.rename({"entity_id": "s1_id", "w": "ws", "a": "as", "hn": "hs"}), on="s1_id")
                  .join(info.rename({"entity_id": "q_id", "w": "wq", "a": "aq", "hn": "hq"}), on="q_id"))
        rej = rej.filter((pl.col("ws").list.set_intersection("wq").list.len() == 0) & (pl.col("wq").list.len() == 1)
                         & (pl.col("wq").list.first().str.len_chars() >= 5))
        rej = rej.with_columns(pl.Series("sim", cpdist(rej["ws"].list.join("").to_list(), rej["wq"].list.first().to_list(),
                                                        scorer=fuzz.ratio, workers=-1))).filter(pl.col("sim") < 50)
        rej = rej.with_columns(pl.Series("asim", cpdist(rej["as"].to_list(), rej["aq"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
        same = rej.filter((pl.col("hs") == pl.col("hq")) & (pl.col("asim") >= 90))
        print(f"{c}: {same.height} rejected brand copies at the same address")
        out.append(same.select("s1_id", "q_id", "prob", pl.lit(c).alias("country")))
    return pl.concat(out)

def fr_address_words(r):
    return r.select("entity_id",
        nrm("business_name").str.replace_all(r"[^\w]+", " ").str.strip_chars().str.split(" ").alias("w"),
        pl.col("business_address").str.extract(r"(?:^|\D)(\d{1,5})(?:\D|$)", 1).alias("hn"),
        nrm("business_address").str.replace_all(r"[^a-z0-9]+", " ").str.replace_all(FR_AREA, " ").str.replace_all(STREET, " ")
            .str.replace_all(r"\s+", " ").str.strip_chars().alias("a"))

def fr_filler(fr_scores, src):
    """France records whose best S1 is unaccepted (prob > 0.3), adding a filler word, same house number and address."""
    s1 = v.countries()
    te = pl.read_parquet(fr_scores + "/test_scores.parquet").join(s1.rename({"entity_id": "s1_id"}), on="s1_id").filter(pl.col("country") == "France")
    best = te.sort("prob", descending=True).unique("q_id", keep="first")
    rej = best.join(pairs(rd(src + "/matching_results.tsv")).select("q_id"), on="q_id", how="anti").filter(pl.col("prob") > 0.3)
    w = fr_address_words(raw())
    rej = (rej.join(w.rename({"entity_id": "s1_id", "w": "ws", "hn": "hs", "a": "as"}), on="s1_id")
              .join(w.rename({"entity_id": "q_id", "w": "wq", "hn": "hq", "a": "aq"}), on="q_id"))
    added = pl.lit(False)
    for x in FILL:
        added = added | (pl.col("wq").list.contains(x) & ~pl.col("ws").list.contains(x))
    f = rej.filter(added.fill_null(False) & (pl.col("hs") == pl.col("hq")))
    f = f.with_columns(pl.Series("asim", cpdist(f["as"].to_list(), f["aq"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    return f.filter(pl.col("asim") >= 85)

def fr_relax(fr_scores, t725_dir, src):
    """France pairs accepted at the shared 0.725 threshold (decoy rules still applied) but not at 0.97, whose record
    has no second S1 scoring above 0.3."""
    fr = france_ids()
    ex = lambda d: pairs(d.filter(pl.col("source1_entity_id").is_in(fr)))
    cur, t725 = ex(rd(src + "/matching_results.tsv")), ex(rd(t725_dir + "/matching_results.tsv"))
    cand = t725.join(cur, on=["s1_id", "q_id"], how="anti").join(cur.select("q_id"), on="q_id", how="anti")
    sc = pl.read_parquet(fr_scores + "/test_scores.parquet")
    sec = (sc.sort("prob", descending=True).with_columns(r=pl.col("prob").rank("ordinal", descending=True).over("q_id"))
             .filter(pl.col("r") == 2).select("q_id", pl.col("prob").alias("p2")))
    cand = cand.join(sec, on="q_id", how="left").with_columns(pl.col("p2").fill_null(0.0))
    return cand.filter(pl.col("p2") <= 0.3)

def main(ui, fr, dst):
    s = dst + "/steps/"
    # 1. France decoy rules (0.97), merge; brand copies are measured against the US-trimmed file the run used then
    v.fraggr(fr, s + "fr097", "0.97")
    v.merge(ui, s + "fr097", s + "v19")
    v.ustrim(ui, s + "v19", s + "v20", "0.048")
    brand = brand_rejected(ui, fr, s + "v20")
    # 2. no US trim; brand copies with prob > 0.5 accepted
    add_pairs(s + "v19", brand.filter(pl.col("prob") > 0.5), s + "v27", "brand copies (all countries)")
    # 3. France: no Groupe/Fils flag, French filler at the shared threshold, France brand copies
    v.fraggr(fr, s + "fr097_fill", "0.97", "0", "1")
    take_france(s + "v27", s + "fr097_fill", s + "v33a")
    add_pairs(s + "v33a", brand.filter((pl.col("country") == "France") & (pl.col("prob") > 0.5)), s + "v33", "France brand copies")
    # 4. France filler additions at the same address
    add_pairs(s + "v33", fr_filler(fr, s + "v33"), s + "v34", "France filler at the same address")
    # 5. France unflagged pairs between 0.725 and 0.97 without a competing S1
    v.fraggr(fr, s + "fr0725_fill", "0.725", "0", "1")
    add_pairs(s + "v34", fr_relax(fr, s + "fr0725_fill", s + "v34"), dst, "France unflagged 0.725-0.97, no competitor")
    if os.environ.get("KEEP_STEPS") is None:
        shutil.rmtree(s)

if __name__ == "__main__":
    main(*sys.argv[1:4])
