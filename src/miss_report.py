"""Which true train pairs does retrieval miss, and why? Writes a category breakdown and a sample of misses."""
import os
import polars as pl

W = os.environ.get("WORK_DIR", "work/")
O = os.environ.get("OUT_DIR", "output/")

def main():
    os.makedirs(O, exist_ok=True)
    gt = pl.read_parquet(W + "train_gt.parquet")
    truth = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "")
               .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("q_id")))
    cand = pl.scan_parquet(W + "train_cand/*.parquet").select("s1_id", "q_id").collect()
    miss = truth.join(cand, on=["s1_id", "q_id"], how="anti")
    print("true pairs", truth.height, "missed", miss.height, round(miss.height / truth.height, 4), flush=True)
    cols = ["entity_id", "country", "business_name", "business_address", "name_n", "addr_n"]
    s1 = pl.read_parquet(W + "train_s1.parquet", columns=cols)
    q = pl.concat([pl.read_parquet(f"{W}train_s{i}.parquet", columns=cols) for i in (2, 3)])
    m = (miss.join(s1, left_on="s1_id", right_on="entity_id").join(q, left_on="q_id", right_on="entity_id", suffix="_q"))
    tok = lambda c: pl.col(c).str.split(" ").list.unique()
    m = m.with_columns(
        pl.col("business_name_q").str.contains(r"[^\x00-\x7F]").alias("q_native"),
        pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("s1_native"),
        (tok("name_n").list.set_intersection(tok("name_n_q")).list.len()
         / tok("name_n").list.set_union(tok("name_n_q")).list.len()).alias("name_jac"),
        (tok("addr_n").list.set_intersection(tok("addr_n_q")).list.len()
         / tok("addr_n").list.set_union(tok("addr_n_q")).list.len()).alias("addr_jac"),
        (pl.col("addr_n").str.extract(r"\b(\d{5,6})\b") == pl.col("addr_n_q").str.extract(r"\b(\d{5,6})\b")).alias("zip_eq"),
        # how crowded is this name: number of S1 records with exactly the same normalised name
        pl.col("name_n").alias("_nm"))
    crowd = s1.group_by("name_n").len().rename({"len": "name_crowd"})
    m = m.join(crowd, left_on="_nm", right_on="name_n", how="left").drop("_nm")
    m = m.with_columns(
        pl.when(pl.col("q_native") | pl.col("s1_native")).then(pl.lit("native_script"))
          .when(pl.col("name_n") == pl.col("name_n_q")).then(pl.lit("same_name_crowded"))
          .when(pl.col("name_jac") == 0).then(pl.lit("no_name_token_overlap"))
          .when(pl.col("name_jac") < 0.5).then(pl.lit("low_name_overlap"))
          .otherwise(pl.lit("high_name_overlap")).alias("cat"))
    for c in ("India", "US"):
        mc = m.filter(pl.col("country") == c)
        print(f"\n== {c}: missed {mc.height}")
        print(mc.group_by("cat").agg(pl.len(), pl.col("addr_jac").mean().round(3).alias("addr_jac"),
                                     pl.col("zip_eq").mean().round(3).alias("zip_eq"),
                                     pl.col("name_crowd").median().alias("crowd_med")).sort("len", descending=True))
    keep = ["country", "cat", "name_jac", "addr_jac", "zip_eq", "name_crowd", "business_name", "business_name_q",
            "business_address", "business_address_q", "name_n", "name_n_q", "addr_n", "addr_n_q"]
    (m.select(keep).group_by("country", "cat").map_groups(lambda g: g.sample(min(40, g.height), seed=1))
      .sort("country", "cat").write_csv(O + "misses_sample.tsv", separator="\t"))
    print("wrote", O + "misses_sample.tsv")

if __name__ == "__main__":
    main()
