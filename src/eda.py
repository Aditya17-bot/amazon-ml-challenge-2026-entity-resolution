import polars as pl, time
t=time.time()
D="dataset/"
def rd(p): return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
s1=rd(D+"train/train_source1.tsv"); s2=rd(D+"train/train_source2.tsv"); s3=rd(D+"train/train_source3.tsv")
gt=rd(D+"train/train_ground_truth.tsv")
t1=rd(D+"test/test_source1.tsv"); t2=rd(D+"test/test_source2.tsv"); t3=rd(D+"test/test_source3.tsv")
print("load",time.time()-t)
for n,d in [("s1",s1),("s2",s2),("s3",s3),("t1",t1),("t2",t2),("t3",t3)]:
    print(n,d.shape,d["country"].value_counts().sort("count",descending=True).head(6).rows(), "empty addr",(d["business_address"]=="").sum(),"empty name",(d["business_name"]=="").sum())
pass
g=gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
g=g.with_columns(pl.col("m").list.len().alias("k"), pl.col("m").list.eval(pl.element().str.starts_with("S2")).list.sum().alias("k2"))
g=g.with_columns((pl.col("k")-pl.col("k2")).alias("k3"))
print("k dist",g["k"].value_counts().sort("k").head(15).rows())
print("k2 dist",g["k2"].value_counts().sort("k2").head(10).rows())
print("k3 dist",g["k3"].value_counts().sort("k3").head(10).rows())
ex=g.explode("m").drop_nulls("m")
print("matched pairs",ex.height,"unique matched",ex["m"].n_unique(), "s2 total",s2.height,"s3 total",s3.height)
# are all s2/s3 records matched to something?
allm=set(ex["m"].to_list())
print("frac s2 matched", s2["entity_id"].is_in(list(allm)).mean(), "frac s3 matched", s3["entity_id"].is_in(list(allm)).mean())
# per country singleton rate
gg=g.join(s1.select("entity_id","country"),left_on="source1_entity_id",right_on="entity_id")
print(gg.group_by("country").agg(pl.len(),(pl.col("k")==0).mean().alias("single"),pl.col("k").mean().alias("meank")).rows())
# examples
s23=pl.concat([s2,s3]).rename({"entity_id":"m"})
smp=ex.join(s1,left_on="source1_entity_id",right_on="entity_id").join(s23,on="m",suffix="_m")
import random
for r in smp.sample(40,seed=1).rows(named=True):
    print(r["country"],"|",r["business_name"],"|",r["business_address"],"  ==>  ",r["business_name_m"],"|",r["business_address_m"],"|",r["country_m"])
print("country mismatch rate", (smp["country"]!=smp["country_m"]).mean())
print("singletons examples:")
for r in gg.filter(pl.col("k")==0).join(s1,left_on="source1_entity_id",right_on="entity_id").head(10).rows(named=True):
    print(r["business_name"],"|",r["business_address"])
