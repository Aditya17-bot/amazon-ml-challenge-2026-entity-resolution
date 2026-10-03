"""Assemble the final submission zip.
usage: python package/make_zip.py <team_name> <output_dir_with_both_tsvs>
"""
import os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_FILES = ["prep.py", "learn_native.py", "block.py", "eval_block.py", "miss_report.py", "bench_views.py", "features.py", "train.py", "predict.py",
             "xenc.py", "group.py", "variants.py", "final.py", "join_xenc.py", "kaggle_run.py", "validate_submission.py"]

def main(team, out_dir):
    dst = os.path.join(ROOT, f"{team}_submission.zip")
    code = "code/business_entity_resolution/"
    items = [(os.path.join(out_dir, f), "output/" + f) for f in ("matching_results.tsv", "candidate_pairs.tsv")]
    items += [(os.path.join(ROOT, "src", f), code + "src/" + f) for f in SRC_FILES]
    items += [(os.path.join(ROOT, "package", f), code + f) for f in ("README.md", "requirements.txt")]
    items += [(os.path.join(ROOT, "native_map_learned.tsv"), code + "native_map_learned.tsv"),
              (os.path.join(ROOT, "package", "Documentation_template.md"), "Documentation_template.md")]
    for p, _ in items:
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, a in items:
            z.write(p, a)
            print("  +", a)
    print(dst, f"{os.path.getsize(dst) / 1e6:.0f} MB")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
