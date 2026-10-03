"""Kaggle runner for the entity-resolution pipeline. MODE is set per kernel: train | test | predict."""
import glob, os, shutil, subprocess, sys, time

MODE = "train"
T0 = time.time()

def sh(cmd, **kw):
    print(f"\n$ {cmd}   [{(time.time()-T0)/60:.1f} min]", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")

sh("pip install -q polars==1.36.1 sparse_dot_topn==1.2.0 rapidfuzz==3.13.0 Unidecode==1.4.0")

def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise SystemExit(f"not found in /kaggle/input: {pattern}")
    return hits[0]

SRC = "/kaggle/temp/src/"
WORK = "/kaggle/temp/work/"
OUT = "/kaggle/working/"
os.makedirs(SRC, exist_ok=True); os.makedirs(WORK, exist_ok=True)
code_dir = os.path.dirname(find("block.py"))
for f in glob.glob(code_dir + "/*"):
    shutil.copy(f, SRC)
shutil.copy(SRC + "native_map_learned.tsv", WORK)

env = dict(os.environ, WORK_DIR=WORK, OUT_DIR=OUT + "output/", FRIENDS_DIR=SRC + "friends/", PYTHONUNBUFFERED="1",
           PAIRS_OUT=OUT + "test_pairs.parquet")
if MODE in ("train", "test", "analyze", "xenc"):
    split = "test" if MODE == "test" else "train"
    env["DATA_DIR"] = os.path.dirname(os.path.dirname(find(f"{split}_source1.tsv"))) + "/"
run = lambda cmd: sh(f"set -o pipefail; cd {SRC} && python {cmd} 2>&1 | tee -a {OUT}log.txt", env=env, executable="/bin/bash")

if MODE == "train":
    run("prep.py train")
    run("block.py train 10")
    run("eval_block.py")
    run("features.py train 5")
    run("train.py 5 4")
    for f in ("lgb.txt", "lgb_block.txt", "model_meta.json"):
        shutil.copy(WORK + f, OUT)
elif MODE == "test":
    run("prep.py test")
    run("block.py test 10")
    run("features.py test 10")
    shutil.copytree(WORK + "test_feats", OUT + "test_feats")
    import polars as pl  # installed above; keep only id columns to save output space
    for i in (1, 2, 3):
        pl.read_parquet(WORK + f"test_s{i}.parquet", columns=["entity_id"]).write_parquet(OUT + f"test_s{i}.parquet")
elif MODE == "analyze":
    run("prep.py train")
    run("bench_views.py")
elif MODE == "xenc":
    # inputs: pair dumps of the train kernel and the scored test pairs of the predict kernel
    for f in ("xenc_train.parquet", "val_preds.parquet", "test_pairs.parquet"):
        shutil.copy(find(f), WORK)
    run("prep.py train test")
    run("xenc.py")
    shutil.copy(find("candidate_pairs.tsv"), OUT + "output/")
    test_dir = os.path.dirname(find("test_source1.tsv"))
    sh(f"python {SRC}validate_submission.py --matching {OUT}output/matching_results.tsv "
       f"--candidate {OUT}output/candidate_pairs.tsv --test-dir {test_dir}")
    shutil.copy(WORK + "xenc_meta.json", OUT)
elif MODE == "predict":
    # inputs: outputs of the train kernel (model) and the test kernel (features + id tables)
    for f in ("lgb.txt", "lgb_block.txt", "model_meta.json", "test_s1.parquet", "test_s2.parquet", "test_s3.parquet"):
        shutil.copy(find(f), WORK)
    os.symlink(os.path.dirname(find("test_feats/*.parquet")), WORK + "test_feats")
    run("predict.py")
    data_dir = os.path.dirname(os.path.dirname(find("test_source1.tsv")))
    sh(f"python {SRC}validate_submission.py --matching {OUT}output/matching_results.tsv "
       f"--candidate {OUT}output/candidate_pairs.tsv --test-dir {data_dir}/test")
print(f"\nALL DONE {MODE} in {(time.time()-T0)/60:.1f} min", flush=True)
