"""Pair features for (S2/S3 record, S1 record) candidates."""
import os, sys, glob, time, gc
import numpy as np, polars as pl
from rapidfuzz import fuzz, distance
from rapidfuzz.process import cpdist

W = os.environ.get("WORK_DIR", "work/")
LEGAL = ["private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc", "pllc",
         "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "gmbh", "opc", "pc", "pa", "ltd", "pvt", "inc", "corp"]
LEGAL_RE = r"\b(" + "|".join(LEGAL) + r")\b"

def load_text(split, country):
    cols = ["entity_id", "name_n", "addr_n"]
    rd = lambda i: pl.scan_parquet(f"{W}{split}_s{i}.parquet").filter(pl.col("country") == country).select(cols).collect()
    s1 = rd(1)
    q = pl.concat([rd(2), rd(3)])
    def enrich(d):
        return d.with_columns(
            pl.col("name_n").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("core"),
            pl.col("addr_n").str.extract_all(r"\b\d+\b").alias("nums"),
        ).with_columns(
            pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().alias("ctok"),
            pl.col("nums").list.first().str.slice(0, 9).cast(pl.Int64, strict=False).alias("hn"),
        ).with_columns(
            pl.col("nums").list.eval(pl.element().filter(pl.element().str.len_chars().is_in([5, 6]))).list.last().alias("zip"),
        )
    return enrich(s1), enrich(q)

def rank_feats(c, s1agg):
    """Candidate-set structure features. c: one chunk (whole queries); s1agg: per-S1 stats over the country."""
    c = c.with_columns((pl.col("sc_w") + pl.col("sc_c") + pl.col("sc_l")).alias("sc"))
    c = c.with_columns(
        pl.col("sc").rank("ordinal", descending=True).over("q_id").alias("rk_q"),
        (pl.col("sc") - pl.col("sc").sort(descending=True).slice(1, 1).first().over("q_id").fill_null(0)).alias("gap_q2"),
        (pl.col("sc").max().over("q_id") - pl.col("sc")).alias("gap_qbest"),
        pl.len().over("q_id").alias("n_q"),
    ).join(s1agg, on="s1_id", how="left")
    return c.with_columns((pl.col("smax") - pl.col("sc")).alias("gap_sbest")).drop("smax")

def text_feats(p, s1, q):
    p = (p.join(q.rename(lambda x: x + "_q" if x != "entity_id" else "q_id"), on="q_id", how="left")
          .join(s1.rename(lambda x: x + "_s" if x != "entity_id" else "s1_id"), on="s1_id", how="left"))
    a, b = p["name_n_q"].to_list(), p["name_n_s"].to_list()
    ca, cb = p["core_q"].to_list(), p["core_s"].to_list()
    aa, ab = p["addr_n_q"].to_list(), p["addr_n_s"].to_list()
    f = {}
    kw = dict(workers=-1, dtype=np.float32)
    f["n_ratio"] = cpdist(a, b, scorer=fuzz.ratio, **kw)
    f["n_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, **kw)
    f["n_tsort"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, **kw)
    f["n_partial"] = cpdist(a, b, scorer=fuzz.partial_ratio, **kw)
    f["c_ratio"] = cpdist(ca, cb, scorer=fuzz.ratio, **kw)
    f["c_tset"] = cpdist(ca, cb, scorer=fuzz.token_set_ratio, **kw)
    f["c_jw"] = cpdist(ca, cb, scorer=distance.JaroWinkler.normalized_similarity, **kw)
    # repeated/reordered words: compare the de-duplicated sorted core tokens
    da, db = p["ctok_q"].list.join(" ").to_list(), p["ctok_s"].list.join(" ").to_list()
    f["d_ratio"] = cpdist(da, db, scorer=fuzz.ratio, **kw)
    f["a_tset"] = cpdist(aa, ab, scorer=fuzz.token_set_ratio, **kw)
    f["a_tsort"] = cpdist(aa, ab, scorer=fuzz.token_sort_ratio, **kw)
    f["a_partial"] = cpdist(aa, ab, scorer=fuzz.partial_token_set_ratio, **kw)
    p = p.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    p = p.with_columns(
        pl.col("name_n_q").str.len_chars().alias("len_nq"), pl.col("name_n_s").str.len_chars().alias("len_ns"),
        pl.col("addr_n_q").str.len_chars().alias("len_aq"), pl.col("addr_n_s").str.len_chars().alias("len_as"),
        (pl.col("core_q") == pl.col("core_s")).cast(pl.Int8).alias("core_eq"),
        pl.when(pl.col("zip_q").is_null() | pl.col("zip_s").is_null()).then(-1)
          .otherwise((pl.col("zip_q") == pl.col("zip_s")).cast(pl.Int8)).alias("zip_eq"),
        pl.col("nums_q").list.set_intersection("nums_s").list.len().alias("num_inter"),
        pl.col("nums_q").list.len().alias("num_q"), pl.col("nums_s").list.len().alias("num_s"),
        pl.col("q_id").str.starts_with("S2").cast(pl.Int8).alias("is_s2"),
        # extra / missing core words, exact name, first house number agreement
        pl.col("ctok_q").list.set_difference("ctok_s").list.len().alias("w_extra"),
        pl.col("ctok_s").list.set_difference("ctok_q").list.len().alias("w_miss"),
        pl.col("ctok_q").list.set_intersection("ctok_s").list.len().alias("w_common"),
        (pl.col("name_n_q") == pl.col("name_n_s")).cast(pl.Int8).alias("n_eq"),
        pl.when(pl.col("hn_q").is_null() | pl.col("hn_s").is_null()).then(-1)
          .otherwise((pl.col("hn_q") - pl.col("hn_s")).abs().clip(0, 1000)).alias("hn_diff"),
    )
    drop = [c for c in p.columns if c.endswith(("_q", "_s")) and c.split("_")[0] in ("name", "addr", "core", "nums", "zip", "country", "ctok", "hn")]
    return p.drop(drop)

def in_sample(col, frac10):
    return (col.hash(seed=11) % 10) < frac10

def build(split, frac10=10):
    """frac10: keep S1 entities in this many tenths (train subsample to save time/memory)."""
    t = time.time()
    out_dir = f"{W}{split}_feats/"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{W}{split}_cand/*.parquet"))
    for c_name in sorted({os.path.basename(p).rsplit("_", 1)[0] for p in files}):
        s1agg = (pl.scan_parquet(f"{W}{split}_cand/{c_name}_*.parquet")
                   .group_by("s1_id").agg(pl.len().alias("n_s"), (pl.col("sc_w") + pl.col("sc_c") + pl.col("sc_l")).max().alias("smax"))
                   .collect())
        s1, q = load_text(split, c_name)
        print(c_name, "s1 with cands", s1agg.height, f"{time.time()-t:.0f}s", flush=True)
        for p in [p for p in files if os.path.basename(p).rsplit("_", 1)[0] == c_name]:
            c = rank_feats(pl.read_parquet(p), s1agg)
            if frac10 < 10:
                c = c.filter(in_sample(pl.col("s1_id"), frac10))
            text_feats(c, s1, q).write_parquet(out_dir + os.path.basename(p))
            gc.collect()
        print("  done", c_name, f"{time.time()-t:.0f}s", flush=True)
        del s1, q, s1agg
        gc.collect()

if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "train"
    build(sp, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
