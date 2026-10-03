"""Speed/recall benchmark of blocking settings on a query sample of one country."""
import sys, time
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

W = "work/"
country = sys.argv[1] if len(sys.argv) > 1 else "India"
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
cols = ["entity_id", "country", "name_n", "addr_n"]
s1 = pl.read_parquet(W + "train_s1.parquet", columns=cols).filter(pl.col("country") == country)
q = pl.concat([pl.read_parquet(f"{W}train_s{i}.parquet", columns=cols) for i in (2, 3)]).filter(pl.col("country") == country).sample(nq, seed=0)
gt = pl.read_parquet(W + "train_gt.parquet")
truth = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("q_id"))
           .filter(pl.col("q_id").is_in(q["entity_id"])))
print(country, "s1", s1.height, "q", q.height, "true pairs", truth.height, flush=True)
s1_ids, q_ids = s1["entity_id"], q["entity_id"]

def run(tag, X1, Xq, k=10):
    t = time.time()
    C = sp_matmul_topn(Xq, X1.T.tocsr(), top_n=k, threshold=0.05, sort=True, n_threads=12).tocoo()
    el = time.time() - t
    d = pl.DataFrame({"q_id": q_ids.gather(C.row), "s1_id": s1_ids.gather(C.col), "sc": C.data})
    d = d.with_columns(pl.col("sc").rank("ordinal", descending=True).over("q_id").alias("rk"))
    j = truth.join(d, on=["s1_id", "q_id"], how="left")
    rec = {kk: round(float((j["rk"] <= kk).fill_null(False).mean()), 4) for kk in (1, 3, 5, 10)}
    print(f"{tag}: {el:.0f}s ({el/nq*1e6:.0f}us/q) recall {rec}", flush=True)
    return d

from block import word_doc
words = (lambda d: (d["name_n"] + " " + d["addr_n"]).to_list()) if len(sys.argv) > 4 and sys.argv[4] == "old" else word_doc
names = lambda d: d["name_n"].str.replace_all(" ", "").to_list()
res = {}
for mdf in [int(a) for a in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["300", "1000", "3000"])]:
    v = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32, max_df=mdf, min_df=2)
    X1 = v.fit_transform(words(s1)); Xq = v.transform(words(q))
    res[f"w{mdf}"] = run(f"word max_df={mdf}", X1, Xq)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, max_df=mdf, min_df=2)
    X1 = v.fit_transform(names(s1)); Xq = v.transform(names(q))
    res[f"c{mdf}"] = run(f"char max_df={mdf}", X1, Xq)
    u = pl.concat([res[f"w{mdf}"], res[f"c{mdf}"]]).unique(["q_id", "s1_id"])
    print(f"  union recall {truth.join(u, on=['s1_id','q_id'], how='inner').height/truth.height:.4f} pairs/q {u.height/nq:.1f}", flush=True)
