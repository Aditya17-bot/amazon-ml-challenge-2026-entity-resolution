"""Build small hand-labelling packs for helpers: India native-script words, France tokens."""
import os
import polars as pl
from unidecode import unidecode

W = "work/"
OUT = "friends/"

def native_words():
    """Whitespace words containing non-latin script from India name/address, with counts."""
    frames = []
    for sp in ("train", "test"):
        for s in (1, 2, 3):
            frames.append(pl.scan_parquet(f"{W}{sp}_s{s}.parquet")
                          .filter(pl.col("country") == "India")
                          .select(pl.concat_str(["business_name", "business_address"], separator=" ").alias("t")))
    w = (pl.concat(frames).select(pl.col("t").str.split(" ")).explode("t")
           .with_columns(pl.col("t").str.strip_chars(" ,.;:()[]|-/"))
           .filter(pl.col("t").str.contains(r"[ऀ-෿]"))
           .group_by("t").len().sort("len", descending=True).head(3000).collect())
    return w.select(pl.col("t").alias("native_word"),
                    pl.col("t").map_elements(unidecode, return_dtype=pl.String).alias("machine_guess"),
                    pl.col("len").alias("count"),
                    pl.lit("").alias("english"))

def france_tokens(col, n):
    frames = [pl.scan_parquet(f"{W}test_s{s}.parquet").filter(pl.col("country") == "France").select(col)
              for s in (1, 2, 3)]
    return (pl.concat(frames).select(pl.col(col).str.split(" ").alias("token")).explode("token")
              .filter(pl.col("token").str.len_chars() > 0)
              .group_by("token").len().sort("len", descending=True).head(n).collect()
              .select("token", pl.col("len").alias("count"), pl.lit("").alias("canonical"), pl.lit("").alias("type")))

def france_rows(n=400):
    parts = []
    for s in (1, 2, 3):
        parts.append(pl.scan_parquet(f"{W}test_s{s}.parquet").filter(pl.col("country") == "France")
                     .select("entity_id", "business_name", "business_address").collect().sample(n, seed=s))
    return pl.concat(parts)

if __name__ == "__main__":
    os.makedirs(OUT + "friend1_india", exist_ok=True)
    os.makedirs(OUT + "friend2_france", exist_ok=True)
    native_words().write_csv(OUT + "friend1_india/india_native_words.tsv", separator="\t")
    france_tokens("name_c", 1500).write_csv(OUT + "friend2_france/france_name_tokens.tsv", separator="\t")
    france_tokens("addr_c", 1500).write_csv(OUT + "friend2_france/france_address_tokens.tsv", separator="\t")
    france_rows().write_csv(OUT + "friend2_france/france_sample_rows.tsv", separator="\t")
    print("done")
