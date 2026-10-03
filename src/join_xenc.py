"""Join the scored pairs of several cross-encoder runs into one file for group.py (encoder ensemble).

  python join_xenc.py <dst_dir> <tag>=<xenc_out_dir> [<tag>=<xenc_out_dir> ...]

Each xenc.py output dir holds val_preds_x.parquet and test_pairs_x.parquet with the encoder score in column "x";
it is renamed x<tag> (e.g. x2..x6) and the files are inner-joined on (s1_id, q_id). group.py uses every x* column.
Same logic as the GROUP_XENC branch of kaggle_run.py.
"""
import os, sys
import polars as pl

def main(dst, specs):
    os.makedirs(dst, exist_ok=True)
    for f in ("val_preds_x.parquet", "test_pairs_x.parquet"):
        m = None
        for spec in specs:
            tag, d = spec.split("=", 1)
            x = pl.read_parquet(os.path.join(d, f)).rename({"x": "x" + tag})
            m = x if m is None else m.join(x.select("s1_id", "q_id", "x" + tag), on=["s1_id", "q_id"], how="inner")
            print(f, tag, x.height, "->", m.height, flush=True)
        m.write_parquet(os.path.join(dst, f))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
