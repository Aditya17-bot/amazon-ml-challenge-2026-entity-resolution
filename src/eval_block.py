"""Recall of candidate set vs ground truth, overall, per country and by view rank."""
import os
import polars as pl
W = os.environ.get("WORK_DIR", "work/")
gt = pl.read_parquet(W + "train_gt.parquet")
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))
c = pl.scan_parquet(W + "train_cand/*.parquet").select("q_id", "s1_id", "rkw", "rkc", "rkl").collect()
nq = pl.concat([pl.scan_parquet(f"{W}train_s{i}.parquet").select("entity_id") for i in (2, 3)]).select(pl.len()).collect().item()
print("cand rows", c.height, "per query", round(c.height / nq, 2), "true pairs", pairs.height)
j = pairs.join(c, on=["s1_id", "q_id"], how="left")
for col in ["rkw", "rkc", "rkl"]:
    print(col, "recall@k:", {k: round(float((j[col] <= k).fill_null(False).mean()), 4) for k in (1, 2, 3, 5, 10)})
print("union recall", round(float(j["rkw"].is_not_null().mean()), 4))
s1 = pl.read_parquet(W + "train_s1.parquet", columns=["entity_id", "country"])
print(j.join(s1, left_on="s1_id", right_on="entity_id").group_by("country").agg(pl.col("rkw").is_not_null().mean()).rows())
