"""A/B blocking variants on full S1 per country, queries = true matches of a 5% S1 sample (train)."""
import os, time
import polars as pl
import block

W = os.environ.get("WORK_DIR", "work/")
B = W + "bench/"

def setup():
    os.makedirs(B, exist_ok=True)
    gt = pl.read_parquet(W + "train_gt.parquet")
    cols = ["entity_id", "country", "name_n", "addr_n"]
    s1 = pl.read_parquet(W + "train_s1.parquet", columns=cols)
    samp = s1.filter(pl.col("entity_id").hash(seed=5) % 20 == 0)["entity_id"]
    pairs = (gt.filter(pl.col("source1_entity_id").is_in(samp.implode()))
               .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))
    q = pl.concat([pl.scan_parquet(f"{W}train_s{i}.parquet").select(cols)
                     .filter(pl.col("entity_id").is_in(pairs["q_id"].implode())).collect() for i in (2, 3)])
    s1.write_parquet(B + "b_s1.parquet"); q.write_parquet(B + "b_s2.parquet"); q.head(0).write_parquet(B + "b_s3.parquet")
    return pairs.join(s1.select("entity_id", "country"), left_on="s1_id", right_on="entity_id")

def run(pairs, name, views, mindf, keep=5):
    orig = block.vectorizer
    def v(view):
        x = orig(view)
        if view == "w":
            x.set_params(min_df=mindf)
        return x
    block.VIEWS, block.vectorizer, block.W, block.KEEP = views, v, B, keep
    d = f"{B}{name}/"
    os.makedirs(d, exist_ok=True)
    t = time.time()
    for c in ("India", "US"):
        block.run_country("b", c, 10, d)
    block.vectorizer = orig
    cand = pl.read_parquet(d + "*.parquet")
    r = pairs.join(cand.with_columns(pl.lit(True).alias("hit")), on=["s1_id", "q_id"], how="left")
    by = r.group_by("country").agg(pl.col("hit").fill_null(False).mean().round(4)).sort("country").rows()
    print(f"### {name}: recall {r['hit'].fill_null(False).mean():.4f} {by} per q {cand.height / cand['q_id'].n_unique():.2f} "
          f"{time.time() - t:.0f}s", flush=True)

if __name__ == "__main__":
    pairs = setup()
    print("bench pairs", pairs.height, flush=True)
    run(pairs, "wcl_keep5", ("w", "c", "l"), 1)
    run(pairs, "wcl_keep10", ("w", "c", "l"), 1, keep=10)
    run(pairs, "wcla_keep5", ("w", "c", "l", "a"), 1)
