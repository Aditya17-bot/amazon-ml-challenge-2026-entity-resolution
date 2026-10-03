"""Stage 3: cross-encoder re-scoring of the pruned candidate pairs, combined with the LightGBM matcher.

1. fine-tune a small pretrained cross-encoder (MiniLM, Apache-2.0) on train pairs: "name | address" of the S2/S3
   record vs the S1 record -> match / no match
2. score the held-out train pairs and all test candidate pairs
3. combiner: LightGBM on [matcher prob, cross-encoder prob, their ranks/gaps within the record's candidates],
   trained on held-out pairs (cross-fitted by S1 id to tune the threshold), applied to test
Inputs (WORK_DIR): xenc_train.parquet, val_preds.parquet, test_pairs.parquet, {train,test}_s{1,2,3}.parquet
Outputs (OUT_DIR): matching_results.tsv (candidate_pairs.tsv is unchanged: the same pairs are scored)
"""
import os, sys, time, json, math
import numpy as np, polars as pl, torch, lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train import gt_pairs, decide, macro_f05, is_valid
from features import in_sample

W = os.environ.get("WORK_DIR", "work/")
O = os.environ.get("OUT_DIR", "output/")
MODEL = os.environ.get("XENC_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
MAXLEN = 96
EPOCHS = int(os.environ.get("XENC_EPOCHS", "1"))
FRAC10 = int(os.environ.get("FRAC10", "5"))  # must match the train run
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

def log(*a):
    print(f"[{(time.time() - T0) / 60:.1f} min]", *a, flush=True)

def text_table(split):
    cols = ["entity_id", "name_n", "addr_n"]
    d = pl.concat([pl.read_parquet(f"{W}{split}_s{i}.parquet", columns=cols) for i in (1, 2, 3)])
    return d.select("entity_id", (pl.col("name_n") + " | " + pl.col("addr_n")).alias("t"))

def attach(pairs, txt):
    return (pairs.join(txt.rename({"entity_id": "q_id", "t": "tq"}), on="q_id", how="left")
                 .join(txt.rename({"entity_id": "s1_id", "t": "ts"}), on="s1_id", how="left")
                 .with_columns(pl.col("tq", "ts").fill_null("")))

def batches(a, b, bs):
    for i in range(0, len(a), bs):
        yield i, a[i:i + bs], b[i:i + bs]

class Amp(torch.nn.Module):
    """fp16 autocast inside forward, so it also applies in DataParallel replica threads; returns the match logit."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, **kw):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            return self.m(**kw).logits[:, 0].float()

def finetune(model, tok, a, b, y, bs=128, lr=4e-5):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    per_epoch = math.ceil(len(a) / bs)
    steps = per_epoch * EPOCHS  # one warmup + linear decay over all epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(a))
        ea, eb, ey = [a[i] for i in perm], [b[i] for i in perm], y[perm]
        for n, (i, xa, xb) in enumerate(batches(ea, eb, bs)):
            enc = tok(xa, xb, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
            logit = NET(**enc)
            loss = lossf(logit, torch.tensor(ey[i:i + bs], dtype=torch.float32, device=DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if n % 2000 == 0:
                log(f"epoch {ep} step {ep * per_epoch + n}/{steps} loss {loss.item():.4f}")

@torch.no_grad()
def score(model, tok, a, b, bs=1024):
    model.eval()
    # length-sorted batches: little padding per batch; results scattered back to input order
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    a, b = [a[k] for k in order], [b[k] for k in order]
    out = np.empty(len(a), dtype=np.float32)
    for n, (i, xa, xb) in enumerate(batches(a, b, bs)):
        enc = tok(xa, xb, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
        out[order[i:i + bs]] = torch.sigmoid(NET(**enc)).cpu().numpy()
        if n % 2000 == 0:
            log(f"scored {i}/{len(a)}")
    return out

def combo_feats(p):
    """p: s1_id, q_id, prob, x (+ country). Ranks and gaps of both scores within each record's candidates."""
    p = p.with_columns((pl.col("prob") + pl.col("x")).alias("px"))
    f = []
    for c in ("prob", "x", "px"):
        f += [pl.col(c).rank("ordinal", descending=True).over("q_id").alias(f"rk_{c}"),
              (pl.col(c).max().over("q_id") - pl.col(c)).alias(f"gapq_{c}"),
              (pl.col(c) - pl.col(c).sort(descending=True).slice(1, 1).first().over("q_id").fill_null(0)).alias(f"gap2_{c}"),
              (pl.col(c).max().over("s1_id") - pl.col(c)).alias(f"gaps_{c}")]
    f += [pl.len().over("q_id").alias("nq"), pl.len().over("s1_id").alias("ns"),
          (pl.col("q_id").str.starts_with("S2")).cast(pl.Int8).alias("is_s2")]
    return p.with_columns(f)

CF = ["prob", "x", "px"] + [f"{k}_{c}" for c in ("prob", "x", "px") for k in ("rk", "gapq", "gap2", "gaps")] + ["nq", "ns", "is_s2"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())

def main():
    os.makedirs(O, exist_ok=True)
    log("device", DEV, torch.cuda.device_count(), "model", MODEL, "epochs", EPOCHS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(DEV)
    global NET
    NET = torch.nn.DataParallel(Amp(model)) if torch.cuda.device_count() > 1 else Amp(model)

    # 1. fine-tune on training-part pairs
    txt = text_table("train")
    tr = attach(pl.read_parquet(W + "xenc_train.parquet"), txt)
    log("xenc train pairs", tr.height, "pos rate", round(float(tr["y"].mean()), 3))
    finetune(model, tok, tr["tq"].to_list(), tr["ts"].to_list(), tr["y"].to_numpy().astype(np.float32))
    del tr
    model.save_pretrained(W + "xenc_model"); tok.save_pretrained(W + "xenc_model")

    # 2. held-out pairs: cross-encoder score, then cross-fitted combiner to measure and pick the threshold
    truth = gt_pairs()
    va = attach(pl.read_parquet(W + "val_preds.parquet"), txt)
    va = va.with_columns(pl.Series("x", score(model, tok, va["tq"].to_list(), va["ts"].to_list()))).drop("tq", "ts")
    va.select("s1_id", "q_id", "prob", "x").write_parquet(O + "val_preds_x.parquet")  # for group.py
    del txt
    va = combo_feats(va).join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "q_id"], how="left").with_columns(pl.col("y").fill_null(0))
    s1 = pl.read_parquet(W + "train_s1.parquet", columns=["entity_id", "country"]).filter(
        is_valid(pl.col("entity_id")) & in_sample(pl.col("entity_id"), FRAC10))
    vtruth = truth.filter(pl.col("s1_id").is_in(s1["entity_id"].implode()))
    fold = (va["s1_id"].hash(seed=13) % 4).to_numpy()
    oof = np.zeros(va.height, dtype=np.float32)
    X, y = va.select(CF).to_numpy().astype(np.float32), va["y"].to_numpy()
    rounds = []
    for k in range(4):
        trn, tst = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[trn], y[trn]), 2000, valid_sets=[lgb.Dataset(X[tst], y[tst])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[tst] = m.predict(X[tst]); rounds.append(m.best_iteration)
    ts = np.round(np.arange(0.3, 0.96, 0.025), 3)
    res = {}
    for name, col in (("matcher only", va["prob"].to_numpy()), ("cross-encoder only", va["x"].to_numpy()), ("combined", oof)):
        p = va.select("s1_id", "q_id").with_columns(pl.Series("prob", col))
        scs = [macro_f05(decide(p, t), vtruth, s1["entity_id"]) for t in ts]
        res[name] = (float(ts[int(np.argmax(scs))]), max(scs))
        log(f"{name}: best t {res[name][0]} F0.5 {res[name][1]:.4f}")
    t_best = res["combined"][0]
    for c in s1["country"].unique().sort().to_list():
        ids = s1.filter(pl.col("country") == c)["entity_id"]
        p = va.select("s1_id", "q_id").with_columns(pl.Series("prob", oof)).filter(pl.col("s1_id").is_in(ids.implode()))
        log("  ", c, round(macro_f05(decide(p, t_best), vtruth.filter(pl.col("s1_id").is_in(ids.implode())), ids), 4))
    comb = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(rounds) * 1.1))
    comb.save_model(W + "lgb_combiner.txt")
    json.dump({"threshold": t_best, "scores": res, "features": CF}, open(W + "xenc_meta.json", "w"))
    del va, X

    # 3. test: score every candidate pair, combine, decide, write matching_results.tsv
    txt = text_table("test")
    te = attach(pl.read_parquet(W + "test_pairs.parquet").select("s1_id", "q_id", "prob"), txt)
    del txt
    te = te.with_columns(pl.Series("x", score(model, tok, te["tq"].to_list(), te["ts"].to_list()))).drop("tq", "ts")
    te.select("s1_id", "q_id", "prob", "x").write_parquet(O + "test_pairs_x.parquet")  # for group.py
    te = combo_feats(te)
    te = te.select("s1_id", "q_id").with_columns(pl.Series("prob", comb.predict(te.select(CF).to_numpy().astype(np.float32))))
    pred = decide(te, t_best)
    s1t = pl.read_parquet(W + "test_s1.parquet", columns=["entity_id"])
    g = pred.group_by("s1_id").agg(pl.col("q_id").sort().str.join(",").alias("matched_entity_ids"))
    (s1t.join(g, left_on="entity_id", right_on="s1_id", how="left").sort("entity_id")
        .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
        .write_csv(O + "matching_results.tsv", separator="\t", quote_style="never"))
    log("test pairs", te.height, "matches", pred.height, "s1 with match", pred["s1_id"].n_unique(), "/", s1t.height)

if __name__ == "__main__":
    main()
