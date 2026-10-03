"""Train LightGBM pair classifier on train features, tune decision threshold for macro F0.5 on held-out S1 ids."""
import os
import sys, time, json
import numpy as np, polars as pl, lightgbm as lgb
from features import in_sample

W = os.environ.get("WORK_DIR", "work/")
NON_FEAT = {"q_id", "s1_id", "y"}
BLOCK_FEATS = ["sc_w", "rkw", "sc_c", "rkc", "sc_l", "rkl", "sc", "rk_q", "gap_q2", "gap_qbest", "n_q", "n_s", "gap_sbest"]
FRAC10 = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # must match features.py train run
NEG_MOD = int(sys.argv[2]) if len(sys.argv) > 2 else 2  # keep 1/NEG_MOD of training negatives

def gt_pairs():
    gt = pl.read_parquet(W + "train_gt.parquet")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("q_id")))

def is_valid(col):
    return (col.hash(seed=7) % 5) == 0

def decide(p, t):
    """Each S2/S3 record goes to its best S1 if prob > t."""
    best = p.sort("prob", descending=True).unique("q_id", keep="first")
    return best.filter(pl.col("prob") > t).select("s1_id", "q_id")

def decide_by(p, tmap):
    """decide() with a per-country threshold; p needs a country column (the S1 record's country)."""
    best = p.sort("prob", descending=True).unique("q_id", keep="first")
    t = pl.col("country").replace_strict(tmap, default=tmap["_all"], return_dtype=pl.Float64)
    return best.filter(pl.col("prob") > t).select("s1_id", "q_id")

def macro_f05(pred, truth, s1_ids):
    """pred/truth: (s1_id,q_id) pairs; s1_ids: all S1 ids evaluated."""
    tp = pred.join(truth, on=["s1_id", "q_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pred.group_by("s1_id").len().rename({"len": "np"})
    ntrue = truth.group_by("s1_id").len().rename({"len": "nt"})
    d = (pl.DataFrame({"s1_id": s1_ids}).join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
           .join(ntrue, on="s1_id", how="left").fill_null(0))
    d = d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))).alias("f"))
    return float(d["f"].mean())

def error_report(pred, truth, cand):
    """Split validation errors into retrieval misses, model misses and wrong matches; dump samples with text."""
    O = os.environ.get("OUT_DIR", "output/")
    os.makedirs(O, exist_ok=True)
    tp = pred.join(truth, on=["s1_id", "q_id"])
    fp = pred.join(truth, on=["s1_id", "q_id"], how="anti")
    fn = truth.join(pred, on=["s1_id", "q_id"], how="anti")
    fn_c = fn.join(cand.select("s1_id", "q_id"), on=["s1_id", "q_id"], how="semi")
    # wrong matches: is the record a decoy (no true S1 anywhere) or does it belong to another S1?
    fp = fp.join(truth.select("q_id", pl.col("s1_id").alias("true_s1")), on="q_id", how="left")
    print(f"errors: tp {tp.height} | fp {fp.height} (decoy {fp['true_s1'].is_null().sum()}, other S1 {fp['true_s1'].is_not_null().sum()}) "
          f"| fn {fn.height} (in candidates {fn_c.height}, not retrieved {fn.height - fn_c.height})", flush=True)
    cols = ["entity_id", "business_name", "business_address"]
    ids = pl.concat([fp["s1_id"], fp["q_id"], fp["true_s1"].drop_nulls(), fn_c["s1_id"], fn_c["q_id"]]).unique()
    txt = pl.concat([pl.scan_parquet(f"{W}train_s{i}.parquet").select(cols).filter(pl.col("entity_id").is_in(ids.implode())).collect()
                     for i in (1, 2, 3)])
    nm = lambda c, p: txt.rename({"entity_id": c, "business_name": p + "_name", "business_address": p + "_addr"})
    pr = cand.select("s1_id", "q_id", "prob")
    (fp.sample(min(400, fp.height), seed=1).join(pr, on=["s1_id", "q_id"], how="left")
       .join(nm("s1_id", "s1"), on="s1_id", how="left").join(nm("q_id", "q"), on="q_id", how="left")
       .join(nm("true_s1", "true"), on="true_s1", how="left").write_csv(O + "val_false_matches.tsv", separator="\t"))
    (fn_c.sample(min(400, fn_c.height), seed=1).join(pr, on=["s1_id", "q_id"], how="left")
       .join(nm("s1_id", "s1"), on="s1_id", how="left").join(nm("q_id", "q"), on="q_id", how="left")
       .write_csv(O + "val_model_misses.tsv", separator="\t"))

def main():
    t0 = time.time()
    truth = gt_pairs()
    va = is_valid(pl.col("s1_id"))
    # training part: all positives + 1/NEG_MOD of negatives (threshold is tuned on the untouched valid part);
    # sampled while scanning so the full feature table never sits in memory
    f = (pl.scan_parquet(W + "train_feats/*.parquet")
           .join(truth.lazy().with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "q_id"], how="left")
           .with_columns(pl.col("y").fill_null(0))
           .filter(va | (pl.col("y") == 1) | (pl.col("q_id").hash(seed=3) % NEG_MOD == 0))
           .collect())
    feats = [c for c in f.columns if c not in NON_FEAT]
    f = f.with_columns(pl.col(feats).cast(pl.Float32))
    tr = f.filter(~va)
    vl = f.filter(va)
    del f
    print("rows", tr.height, vl.height, "pos rate", float(tr["y"].mean()), f"{time.time()-t0:.0f}s", flush=True)
    dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy(), feature_name=feats, free_raw_data=True)
    dvl = lgb.Dataset(vl.select(feats).to_numpy(), vl["y"].to_numpy(), reference=dtr)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())
    m = lgb.train(params, dtr, 1500, valid_sets=[dvl], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(W + "lgb.txt")
    vl_x = vl
    vl = vl.select("s1_id", "q_id").with_columns(pl.Series("prob", m.predict(vl.select(feats).to_numpy())))
    # evaluate on valid S1 ids that were also in the feature subsample
    s1_all = pl.read_parquet(W + "train_s1.parquet", columns=["entity_id", "country"])
    s1 = s1_all.filter(is_valid(pl.col("entity_id")) & in_sample(pl.col("entity_id"), FRAC10))
    vtruth = truth.filter(pl.col("s1_id").is_in(s1["entity_id"]))
    vl = vl.join(s1_all, left_on="s1_id", right_on="entity_id", how="left")
    ts = np.round(np.arange(0.2, 0.96, 0.05), 2)
    best_t, best = 0.5, -1
    for t in ts:
        sc = macro_f05(decide(vl, t), vtruth, s1["entity_id"])
        print(f"  t={t:.2f} F0.5={sc:.4f}")
        if sc > best:
            best_t, best = float(t), sc
    tmap = {"_all": best_t}  # countries without train data (France) use the global threshold
    for c in s1["country"].unique().sort().to_list():
        ids = s1.filter(pl.col("country") == c)["entity_id"]
        vc, tc = vl.filter(pl.col("country") == c), vtruth.filter(pl.col("s1_id").is_in(ids))
        scs = [macro_f05(decide(vc, t), tc, ids) for t in ts]
        tmap[c] = float(ts[int(np.argmax(scs))])
        print("  ", c, "global t", round(scs[list(ts).index(round(best_t, 2))], 4), "own t", tmap[c], round(max(scs), 4))
    best = macro_f05(decide_by(vl, tmap), vtruth, s1["entity_id"])
    print("per-country thresholds", tmap, "F0.5", round(best, 4), flush=True)
    # stage-1 blocking pruner: small model on retrieval scores/ranks only; its survivors are the candidate set
    mb = lgb.train(dict(params, num_leaves=31), lgb.Dataset(tr.select(BLOCK_FEATS).to_numpy(), tr["y"].to_numpy(),
                   feature_name=BLOCK_FEATS), 300)
    mb.save_model(W + "lgb_block.txt")
    vl = vl.with_columns(pl.Series("pb", mb.predict(vl_x.select(BLOCK_FEATS).to_numpy())))
    vs1 = vl.filter(pl.col("s1_id").is_in(s1["entity_id"]))
    cand_t = 0.0
    for ct in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        kept = vs1.filter(pl.col("pb") >= ct)
        rec = kept.join(vtruth, on=["s1_id", "q_id"]).height / max(vtruth.height, 1)
        sc = macro_f05(decide_by(kept, tmap), vtruth, s1["entity_id"])
        print(f"  block prob>={ct}: per S1 {kept.height / s1.height:.2f}  pair recall {rec:.4f}  F0.5 {sc:.4f}", flush=True)
        if sc >= best - 0.001:
            cand_t = ct
    print("cand_t", cand_t)
    vk = vl.filter(pl.col("pb") >= cand_t)
    error_report(decide_by(vk, tmap), vtruth, vk)
    # pairs for the cross-encoder stage (xenc.py): its training pairs come from the training part only (after the
    # same stage-1 prune); the held-out part, with the matcher's prob, trains the final combiner
    O = os.environ.get("OUT_DIR", "output/")
    trk = tr.select("s1_id", "q_id", "y").filter(pl.Series(mb.predict(tr.select(BLOCK_FEATS).to_numpy()) >= cand_t))
    trk.sample(min(3_000_000, trk.height), seed=1).write_parquet(O + "xenc_train.parquet")
    vk.select("s1_id", "q_id", "prob", "country").write_parquet(O + "val_preds.parquet")
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
    print("importance", [(a, int(b)) for a, b in imp[:15]])
    json.dump({"threshold": best_t, "thresholds": tmap, "valid_f05": best, "features": feats, "block_features": BLOCK_FEATS,
               "cand_t": cand_t}, open(W + "model_meta.json", "w"))
    print("best", best_t, best, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
