"""Score test candidates with the trained model and write output/matching_results.tsv + candidate_pairs.tsv."""
import os, glob, json
import numpy as np, polars as pl, lightgbm as lgb

W = os.environ.get("WORK_DIR", "work/")
O = os.environ.get("OUT_DIR", "output/")

def write_lists(pairs, s1_tab, q_tab, col, path):
    """pairs: int (si, qi) row numbers into s1_tab / q_tab."""
    g = (pairs.unique(["si", "qi"]).join(q_tab, on="qi").group_by("si")
              .agg(pl.col("q_id").sort().str.join(",").alias(col)))
    d = (s1_tab.join(g, on="si", how="left").sort("si").with_columns(pl.col(col).fill_null(""))
               .select(pl.col("s1_id").alias("source1_entity_id"), col))
    d.write_csv(path, separator="\t", quote_style="never")

def main():
    os.makedirs(O, exist_ok=True)
    meta = json.load(open(W + "model_meta.json"))
    m = lgb.Booster(model_file=W + "lgb.txt")
    s1_tab = pl.read_parquet(W + "test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"}).with_row_index("si")
    q_tab = (pl.concat([pl.read_parquet(f"{W}test_s{i}.parquet", columns=["entity_id"]) for i in (2, 3)])
               .rename({"entity_id": "q_id"}).with_row_index("qi"))
    mb = lgb.Booster(model_file=W + "lgb_block.txt")
    ct = float(os.environ.get("CAND_T", meta["cand_t"]))
    parts, n_all = [], 0
    for path in sorted(glob.glob(W + "test_feats/*.parquet")):
        f = pl.read_parquet(path)
        n_all += f.height
        # stage 1 (blocking): prune retrieval candidates with the score/rank-only model
        pb = mb.predict(f.select(meta["block_features"]).cast(pl.Float32).to_numpy())
        f = f.filter(pl.Series(pb >= ct))
        # stage 2 (matching): full model on the surviving candidates
        prob = m.predict(f.select(meta["features"]).cast(pl.Float32).to_numpy()).astype(np.float32)
        country = os.path.basename(path).rsplit("_", 1)[0]  # feature files are {country}_{chunk}.parquet
        t = meta.get("thresholds", {}).get(country, meta["threshold"])
        parts.append(f.select("s1_id", "q_id").with_columns(pl.Series("prob", prob), pl.lit(t, pl.Float32).alias("t"))
                      .join(s1_tab, on="s1_id").join(q_tab, on="q_id").select("si", "qi", "prob", "t"))
    p = pl.concat(parts)
    del parts
    # decision: each S2/S3 record goes to its best S1 if prob > threshold
    best = p.sort("prob", descending=True).unique("qi", keep="first")
    pred = best.filter(pl.col("prob") > pl.col("t"))
    print("retrieved pairs", n_all, "candidates after stage-1 prune", p.height, "cand_t", ct)
    if os.environ.get("PAIRS_OUT"):  # scored candidate pairs for the cross-encoder stage (xenc.py)
        (p.join(s1_tab, on="si").join(q_tab, on="qi").select("s1_id", "q_id", "prob", "t")
          .write_parquet(os.environ["PAIRS_OUT"]))
    write_lists(p, s1_tab, q_tab, "candidate_entity_ids", O + "candidate_pairs.tsv")
    write_lists(pred, s1_tab, q_tab, "matched_entity_ids", O + "matching_results.tsv")
    print("candidates", p.height, "matches", pred.height, "s1 with match", pred["si"].n_unique(), "/", s1_tab.height)

if __name__ == "__main__":
    main()
