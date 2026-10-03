"""Learn native-script word -> English word map from training matches (names aligned word by word)."""
import os
import polars as pl

D = os.environ.get("DATA_DIR", "dataset/")
W = os.environ.get("WORK_DIR", "work/")
NATIVE = r"[ऀ-෿]"
STRIP = " ,.;:()[]|-/#'\""

def rd(p, cols):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=cols)

def words(e):
    return e.str.replace_all(r"[,;:()\[\]|/#\"]", " ").str.strip_chars().str.split(" ").list.eval(
        pl.element().str.strip_chars(STRIP).filter(pl.element() != ""))

def main():
    cols = ["entity_id", "business_name", "business_address"]
    s1 = rd(D + "train/train_source1.tsv", cols)
    q = pl.concat([rd(f"{D}train/train_source{i}.tsv", cols) for i in (2, 3)]).filter(
        pl.col("business_name").str.contains(NATIVE) | pl.col("business_address").str.contains(NATIVE))
    gt = pl.read_parquet(W + "train_gt.parquet")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("q_id")))
    p = (pairs.join(q.rename({"entity_id": "q_id"}), on="q_id")
              .join(s1.rename({"entity_id": "s1_id"}), on="s1_id", suffix="_s"))
    print("native pairs", p.height, flush=True)
    rows = []
    # names: same word count -> positional alignment
    n = p.select(words(pl.col("business_name")).alias("a"), words(pl.col("business_name_s").str.to_lowercase()).alias("b"))
    n = n.filter((pl.col("a").list.len() == pl.col("b").list.len()) & (pl.col("a").list.len() > 0))
    rows.append(n.explode(["a", "b"]))
    # addresses: last comma segment (state) and second-to-last (city/district)
    for i in (-1, -2):
        a = p.select(pl.col("business_address").str.split(",").list.get(i, null_on_oob=True).str.strip_chars().alias("a"),
                     pl.col("business_address_s").str.split(",").list.get(i, null_on_oob=True).str.strip_chars().str.to_lowercase().alias("b"))
        rows.append(a.drop_nulls())
    al = pl.concat(rows).filter(pl.col("a").str.contains(NATIVE) & ~pl.col("b").str.contains(r"[^\x00-\x7F]") & (pl.col("b") != ""))
    c = al.group_by("a", "b").len()
    tot = c.group_by("a").agg(pl.col("len").sum().alias("tot"))
    best = (c.sort("len", descending=True).unique("a", keep="first").join(tot, on="a")
              .with_columns((pl.col("len") / pl.col("tot")).alias("share"))
              .filter((pl.col("len") >= 3) & (pl.col("share") >= 0.6)).sort("tot", descending=True))
    best.write_csv(W + "native_map_learned.tsv", separator="\t")
    print("learned", best.height, "words")
    print(best.head(30))

if __name__ == "__main__":
    main()
