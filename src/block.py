"""Candidate generation: for every S2/S3 record find top-K S1 records (same country) by sparse TF-IDF cosine.

Two views: word tokens of name+address (plus whole-name token), and char 3-grams of name.
Queries are processed in chunks; for each chunk both views are scored, merged (keep pairs ranked <= KEEP
in either view) and written straight to disk, so peak memory stays small.
Output: work/{split}_cand/{country}_{chunk:03d}.parquet with q_id, s1_id, sc_w, rkw, sc_c, rkc.
"""
import os, sys, glob, time, gc
import numpy as np, polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

W = os.environ.get("WORK_DIR", "work/")
KEEP = 5
VIEWS = ("w", "c", "l")
LEGAL_RE = (r"\b(private|limited|incorporated|corporation|company|llc|llp|lp|plc|pllc|sarl|sas|sasu|eurl|sa|sci|snc|"
            r"gmbh|opc|pc|pa|ltd|pvt|inc|corp)\b")

def word_doc(df):
    """name words (n_), address words (a_), plus whole legal-stripped name as one sorted token (k_)."""
    core = (df["name_n"].str.replace_all(LEGAL_RE, " ").str.split(" ")
              .list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join("_"))
    name = df["name_n"].str.replace_all(r"(\S+)", "n_$1")
    addr = df["addr_n"].str.replace_all(r"(\S+)", "a_$1")
    return (name + " " + addr + " k_" + core).to_list()

def pair_doc(df):
    """name word x address word pairs: tells apart same-named businesses by any shared locality/number."""
    t = (df.select(pl.int_range(pl.len()).alias("_i"),
                   pl.col("name_n").str.replace_all(LEGAL_RE, " ").str.split(" ").list.unique().alias("n"),
                   pl.col("addr_n").str.split(" ").list.unique().alias("a"))
           .explode("n").filter(pl.col("n") != "").explode("a").filter(pl.col("a") != "")
           .group_by("_i").agg((pl.col("n") + "|" + pl.col("a")).str.join(" ").alias("d")))
    return pl.DataFrame({"_i": pl.int_range(df.height, eager=True)}).join(t, on="_i", how="left")["d"].fill_null("").to_list()

def view_texts(df, view):
    if view == "w":
        return word_doc(df)
    if view == "l":
        return pair_doc(df)
    if view == "a":
        return df["addr_n"].to_list()
    return df["name_n"].str.replace_all(" ", "").to_list()

class PairVec:
    """Hashed tf-idf for the name x address pair view; buckets in more than max_df S1 docs are dropped."""
    def __init__(self, max_df=2000):
        self.h = HashingVectorizer(token_pattern=r"\S+", n_features=2**24, alternate_sign=False, norm=None, dtype=np.float32)
        self.t = TfidfTransformer(sublinear_tf=True)
        self.max_df = max_df
    def fit_transform(self, docs):
        X = self.h.transform(docs).tocsc()
        df = np.diff(X.indptr)
        self.keep = sp.diags(((df >= 1) & (df <= self.max_df)).astype(np.float32))
        return self.t.fit_transform((X @ self.keep).tocsr()).astype(np.float32)
    def transform(self, docs):
        return self.t.transform(self.h.transform(docs) @ self.keep).astype(np.float32)

def vectorizer(view):
    if view == "w":  # min_df=1: a word/name seen once in S1 is the most telling, never drop it
        return TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32, max_df=6000, min_df=1)
    if view == "l":
        return PairVec()
    if view == "a":  # address only: brand name vs registered name, same premises
        return TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32, max_df=3000, min_df=1)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, max_df=3000, min_df=2)

def topk(A, BT, k, name, n_threads):
    C = sp_matmul_topn(A, BT, top_n=k, threshold=0.05, sort=True, n_threads=n_threads).tocoo()
    d = pl.DataFrame({"qi": C.row.astype(np.int32), "si": C.col.astype(np.int32), f"sc_{name}": C.data.astype(np.float32)})
    return d.with_columns(pl.col(f"sc_{name}").rank("ordinal", descending=True).over("qi").cast(pl.Int8).alias(f"rk{name}"))

def run_country(split, c, k, out_dir, n_threads=os.cpu_count(), chunk=250_000):
    t = time.time()
    cols = ["entity_id", "name_n", "addr_n"]
    rd = lambda i: pl.scan_parquet(f"{W}{split}_s{i}.parquet").filter(pl.col("country") == c).select(cols).collect()
    s1 = rd(1)
    q = pl.concat([rd(2), rd(3)])
    print(split, c, s1.height, q.height, flush=True)
    vecs, BTs = {}, {}
    for view in VIEWS:
        vecs[view] = vectorizer(view)
        BTs[view] = vecs[view].fit_transform(view_texts(s1, view)).T.tocsr()
    s1_ids = s1["entity_id"]
    del s1
    gc.collect()
    total = 0
    for n, lo in enumerate(range(0, q.height, chunk)):
        dst = f"{out_dir}{c}_{n:03d}.parquet"
        if os.path.exists(dst):
            continue
        qc = q.slice(lo, chunk)
        m = None
        for view in VIEWS:
            v = topk(vecs[view].transform(view_texts(qc, view)), BTs[view], k, view, n_threads)
            m = v if m is None else m.join(v, on=["qi", "si"], how="full", coalesce=True)
        m = m.filter(pl.any_horizontal([pl.col(f"rk{v}") <= KEEP for v in VIEWS]))
        m = m.with_columns(pl.col([f"sc_{v}" for v in VIEWS]).fill_null(0.0), pl.col([f"rk{v}" for v in VIEWS]).fill_null(k + 1))
        m = (m.with_columns(qc["entity_id"].gather(m["qi"]).alias("q_id"), s1_ids.gather(m["si"]).alias("s1_id"))
              .select("q_id", "s1_id", *[c for v in VIEWS for c in (f"sc_{v}", f"rk{v}")]))
        m.write_parquet(dst)
        total += m.height
        if n % 4 == 0:
            print(f"   {c} chunk {n} rows {total} {time.time()-t:.0f}s", flush=True)
    print("  done", c, f"{time.time()-t:.0f}s", flush=True)

def run(split, k=10):
    out_dir = f"{W}{split}_cand/"
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(out_dir + "_*"):  # leftovers of the old layout
        os.remove(f)
    countries = pl.scan_parquet(f"{W}{split}_s1.parquet").select("country").unique().collect()["country"].sort().to_list()
    for c in countries:
        run_country(split, c, k, out_dir)
        gc.collect()

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "train", int(sys.argv[2]) if len(sys.argv) > 2 else 10)
