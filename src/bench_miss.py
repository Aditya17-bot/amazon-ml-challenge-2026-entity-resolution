import sys
exec(open("src/bench_block.py").read().split("res = {}")[0])
v = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32, max_df=6000, min_df=2)
X1 = v.fit_transform(words(s1)); Xq = v.transform(words(q))
d = run("w6000", X1, Xq, k=20)
j = truth.join(d, on=["s1_id", "q_id"], how="left").filter(pl.col("sc").is_null())
t1 = pl.read_parquet(W + "train_s1.parquet", columns=["entity_id","business_name","business_address"])
tq = pl.concat([pl.read_parquet(f"{W}train_s{i}.parquet", columns=["entity_id","business_name","business_address"]) for i in (2,3)])
m = j.join(t1, left_on="s1_id", right_on="entity_id").join(tq, left_on="q_id", right_on="entity_id", suffix="_q")
nonasc = m["business_name_q"].str.contains(r"[^\x00-\x7F]").mean()
print("missed", m.height, "frac q name non-ascii", nonasc, "frac s1 name non-ascii", m["business_name"].str.contains(r"[^\x00-\x7F]").mean())
for r in m.sample(25, seed=2).rows(named=True):
    print(r["business_name"], "|", r["business_address"], "\n   => ", r["business_name_q"], "|", r["business_address_q"])
