# Business Entity Resolution: reproduction guide

Pipeline: text normalisation, then TF-IDF candidate retrieval, then pairwise features, then a small LightGBM blocking pruner (scores/ranks only), then the LightGBM matching classifier on the surviving candidates, then assignment of each S2/S3 record to at most one S1 record.
No external data or APIs are used. Every dictionary is either hand-written in code or learned from the training ground truth.

## Setup
```
pip install -r requirements.txt
```
Put the challenge data at `dataset/train/*.tsv` and `dataset/test/*.tsv`. The paths can be overridden with the `DATA_DIR`, `WORK_DIR` and `OUT_DIR` environment variables.

## Run (from this folder)
Stage 1: normalisation, retrieval, matcher (CPU).
```
python src/learn_native.py      # native-script -> English word map learned from train matches (work/native_map_learned.tsv)
python src/prep.py train test   # normalise names/addresses -> work/*.parquet
python src/block.py train 10    # candidate generation, train
python src/eval_block.py        # blocking recall on train
python src/features.py train 5  # pair features on a 50% sample of S1 entities
python src/train.py 5 4         # (all positives + 1/4 of negatives) matcher + blocking pruner, F0.5 threshold + prune cut-off tuned on held-out S1 ids -> work/lgb.txt, lgb_block.txt, model_meta.json
python src/block.py test 10     # candidate generation, test
python src/features.py test 10  # pair features, all test candidates
PAIRS_OUT=work/test_pairs.parquet python src/predict.py   # -> output/candidate_pairs.tsv (+ stage-1 matching_results.tsv), work/test_pairs.parquet
cp output/xenc_train.parquet output/val_preds.parquet work/   # train.py writes the encoder's train/validation pairs to output/
```
Stage 2: five cross-encoder runs (GPU), each fine-tuned on the train pairs and scoring validation + test pairs.
The tag is the name of the score column group.py sees (x2..x6).
```
OUT_DIR=runs/x2/                                                                      python src/xenc.py  # MiniLM-L12, 1 epoch
OUT_DIR=runs/x3/ XENC_EPOCHS=2                                                        python src/xenc.py  # MiniLM-L12, 2 epochs
OUT_DIR=runs/x4/ XENC_EPOCHS=2 XENC_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  python src/xenc.py  # multilingual, 2 epochs
OUT_DIR=runs/x5/ XENC_EPOCHS=2                                                        python src/xenc.py  # 2nd run of x3 (France rebuild pipeline)
OUT_DIR=runs/x6/ XENC_EPOCHS=2 XENC_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  python src/xenc.py  # 2nd run of x4 (France rebuild pipeline)
```
Stage 3: group check / combiner (CPU), two runs: US/India with all five encoders, France with the two rebuild encoders.
```
python src/join_xenc.py work/ 2=runs/x2 3=runs/x3 4=runs/x4 5=runs/x5 6=runs/x6
VAL_FILE=val_preds_x.parquet TEST_FILE=test_pairs_x.parquet OUT_DIR=runs/group_ui/ python src/group.py   # matching_results.tsv, test_scores.parquet, group_meta.json
python src/join_xenc.py work/ 5=runs/x5 6=runs/x6
VAL_FILE=val_preds_x.parquet TEST_FILE=test_pairs_x.parquet OUT_DIR=runs/group_fr/ python src/group.py
cp output/candidate_pairs.tsv runs/group_ui/ && cp output/candidate_pairs.tsv runs/group_fr/
```
Stage 4: final decisions (CPU, about 10 minutes): France decoy rules and the leaderboard-tested additions, see the docstring of `src/final.py`.
```
cd src && WORK_DIR=../work/ python final.py ../runs/group_ui ../runs/group_fr ../output_final && cd ..
python src/validate_submission.py --matching output_final/matching_results.tsv --candidate output_final/candidate_pairs.tsv --test-dir dataset/test
```
`output_final/` then holds the two submitted files. Run with `KEEP_STEPS=1` to keep the intermediate files of each step.

Notes:
- `learn_native.py` reads `work/train_gt.parquet`, which `prep.py` writes. On a fresh checkout, run `prep.py` once, then `learn_native.py`, then `prep.py` again. Alternatively, use the shipped `native_map_learned.tsv`: copy it to `work/`.
- Our US/India scores (x2, x3, x4) come from a first run of stage 1 whose `prep.py` still deleted French legal forms (SAS, SARL, ...); the shipped `prep.py` turns them into tokens. That change only touches France records, so US/India text, candidates and features are the same. France (x5, x6 and the France group run) comes from the shipped code.
- The cross-encoders train on GPU with fp16, so a rerun gives scores that differ in the last digits; the thresholds are re-tuned on validation in every run.
- Our leaderboard file is `output/matching_results.tsv` of the submission zip; `final.py` rebuilds it byte-for-byte from our two group-run outputs (checked).

Resources: stage 1 is memory-bounded (chunked) and needs about 8 GB RAM; with 4 CPU cores it takes about 5 hours (blocking is the dominant cost). Each cross-encoder run takes about 3 hours on one Kaggle GPU.
We ran stages 1-3 as Kaggle notebooks with `src/kaggle_run.py` (one kernel per step; `MODE`, `SUF` and `EXTRA_ENV` set per kernel), and stage 4 locally.

## Files
| file | role |
|---|---|
| `src/prep.py` | lowercasing, transliteration (learned native-script map + unidecode), abbreviation/state-code expansion, per-country token maps |
| `src/learn_native.py` | aligns native-script and English names/addresses of known matches word by word to learn a translation map |
| `src/block.py` | per-country sparse TF-IDF top-k: word view (name/address tokens + whole-name token), char-3-gram name view, and hashed name-word x address-word pair view |
| `src/features.py` | name/address fuzzy similarities (rapidfuzz), number/zip agreement, candidate-rank structure features |
| `src/train.py` | LightGBM matcher (all features) and LightGBM blocking pruner (retrieval scores/ranks only); tunes the match threshold and the largest prune cut-off that keeps macro F0.5 within 0.001 on held-out S1 entities |
| `src/predict.py` | stage 1: pruner drops retrieval pairs below the cut-off; the survivors are exactly `candidate_pairs.tsv`. Stage 2: the matcher scores only those, and each S2/S3 record goes to its best S1 if the probability is above the threshold |
| `src/eval_block.py` | retrieval recall against train ground truth |
| `src/miss_report.py` | analysis only: categorises the true train pairs that retrieval misses |
| `src/bench_views.py` | analysis only: compares retrieval variants (views, `min_df`, keep) on a 5% S1 sample |
| `src/xenc.py` | fine-tunes a cross-encoder on train pairs, scores validation and test pairs, LightGBM combiner of matcher + encoder scores |
| `src/join_xenc.py` | joins the scores of several cross-encoder runs (columns x2..x6) for group.py |
| `src/group.py` | group check: second LightGBM on candidate-group features, decoy-signature features, France threshold calibration and decoy rule |
| `src/variants.py` | post-processing of finished group runs: France decoy rules (`fraggr`), US/India + France merge, US trim, other leaderboard probes |
| `src/final.py` | the exact chain that produced the submitted file from the two group runs |
| `src/kaggle_run.py` | the Kaggle notebook runner we used for stages 1-3 |
| `src/validate_submission.py` | official format validator (copied from the challenge kit) |
| `native_map_learned.tsv` | the learned native-script word map (output of `learn_native.py`) |
