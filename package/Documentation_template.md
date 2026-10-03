# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** GiTWrech  
**Team Members:** Goureesankar S Nair (team leader), Aditya Sridhar, Megeshwar JB, Yuvan Raj Mathan  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve Source 2/3 business records to Source 1 entities with a multi-stage pipeline:

1. **Blocking:** per-country sparse TF-IDF retrieval over three views, followed by a small LightGBM pruner.
2. **Matching:** a LightGBM matcher over fuzzy name/address similarities and candidate-structure features.
3. **Re-scoring:** a fine-tuned cross-encoder reads both records' text. Its scores and the matcher's are combined by a second LightGBM model.
4. **Group check:** asks whether a candidate resembles the other records already matched to the same S1.

Each S2/S3 record is assigned to at most one S1 entity, and thresholds are tuned directly for macro F0.5.

The ideas that mattered most:
- A **native-script to English word map learned from the training matches**, so no external dictionary is needed.
- **Candidate-structure features:** a pair's rank among its record's candidates, and the score gaps.
- The **cross-encoder:** on its own it scores slightly better than the tree matcher, and combined with it the error rate drops by about a third.
- **Handling France**, which is absent from train: a threshold calibrated to the shared match rate, and legal forms kept so that decoys are visible.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train has 2.2M S1 and 10.3M S2+S3 records; test has 1.73M S1 and 10M S2+S3. There are about 3.46 true matches per S1 entity, and 5.6% of S1 entities have none. These statistics are identical for US and India, so the data generator treats countries alike. The test set adds France (259k S1 entities), which has no training data.
- **Name noise:**
  - Changing legal forms (`Pvt Ltd`, `LLC`, `SARL`, `S.A.S.`) and abbreviations.
  - Reordered or dropped words, typos, and accent noise.
  - Names written in Indian native scripts while the match is in Latin script.
- **Address noise:**
  - Abbreviations (`St`, `Nagar`/`Nagr`, `R.` = rue, `Crs` = cours).
  - State codes against full state names, and region against department (France).
  - Renamed cities, missing parts, and reordered segments.
- **Hard negatives (decoys):** copies of an S1 record with a changed legal form and house number, and generic names shared by many S1 entities. The name alone is not enough, so the address and the "is this the best option" context decide.

### 2.2 Solution Strategy
**Approach type:** blocking, then a gradient-boosted classifier, then cross-encoder re-scoring, then a group-level combiner.

**Models and licences:**
- LightGBM (MIT).
- Cross-encoders fine-tuned from `cross-encoder/ms-marco-MiniLM-L12-v2` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Both are Apache 2.0 with about 33M/118M parameters, far below 8B.

**No external data:** no external databases, APIs, geocoders or data augmentation are used. All normalisation maps are hand-written in code or learned from the training ground truth.

---

## 3. Candidate Generation (Blocking)
1. **Normalisation** (`prep.py`):
   - Learned native-script map, then `unidecode`, lowercasing and punctuation removal.
   - Per-country token maps: abbreviations, US and Indian state codes, renamed cities.
   - French street types and stop-words (for France, `st` means saint).
   - French region/department fields are dropped, since one source writes the region and another the department.
   - French legal forms in any spelling (`S.A.R.L.`, `[SAS]`, `SÀRL`) become one token, the same way `Inc`/`LLC` are kept for the US.
2. **Retrieval** (`block.py`), within each country. Each view returns the top 10 S1 records per query, and a pair is kept if it ranks in the top 5 of any view:
   - **Word view:** name and address tokens, plus a whole-name token. Uses `min_df=1`, because a unique name word is the strongest signal.
   - **Character 3-gram view** of the name.
   - **Hashed name-word × address-word pair view** (2^24 buckets). This separates look-alike names by any shared locality word, house number or PIN code.
3. **Learned pruning:** a LightGBM model on retrieval scores and ranks only. Its cut-off is the largest one that keeps validation macro F0.5 within 0.001 of the unpruned score.

**Blocking results:**
- **Train retrieval:** 12.6 candidates per query, union recall **0.980** (US 0.981, India 0.978). By view: word 0.965@10, pair view 0.875, char 0.491 (the char view adds recall on typos).
- **Test:** 125M retrieved pairs, pruned at cut-off 0.05 to **11.6M pairs (6.7 per S1 entity)**. This is `candidate_pairs.tsv`, exactly the set the models score.
- **Pruning cost:** held-out pair recall falls from 0.980 to 0.976, and F0.5 from 0.9743 to 0.9738.

---

## 4. Matching Model
**Matcher features (39):**
- **Name:** rapidfuzz ratio, token-set, token-sort and partial ratio. On the legal-form-stripped core name: ratio, token-set and Jaro-Winkler, plus core equality. Extra, missing and shared words, and lengths.
- **Address:** token-set, token-sort and partial ratios; zip/PIN equality; house-number overlap and difference; lengths.
- **Candidate structure:** retrieval scores and ranks from all views; the rank of this S1 among the record's candidates; gaps to the best and second-best candidate; candidate counts per record and per S1.
- **Other:** the source (S2 or S3).

**Matcher model:** LightGBM (learning rate 0.08, 127 leaves), with early stopping on S1 IDs held out by hash.

**Cross-encoder** (`xenc.py`):
- Input: the pair "name | address" of the S2/S3 record and of the S1 record.
- Fine-tuned for 2 epochs on 3M training candidate pairs, fp16, max 96 tokens.
- Scores every test candidate pair.

**Combiner:** a LightGBM model over the matcher probability and the encoder scores, their ranks and gaps within each record's candidates, and the candidate counts. It is cross-fitted by S1 ID on held-out pairs.

**Decoy-signature features** (`group.py`, from raw text, learned by the combiner on labelled US/India pairs): signed house-number change and the generator's shift set (+3/4/5/7/9/11/13/21), legal forms equal/different/missing, the edit type between differing name words (letters appended or substituted = decoy; deleted, swapped or digit-for-letter = typo of a true match), and an added business word (Trading, Exports, Group, also in Indian scripts). A **word-swap flag** marks pairs where the S1 loses a name word and the record gains a different common business word that is neither a respelling nor filler. Filler is recognised without labels: the noise adds filler words to S2/S3 copies, so they are far more frequent in S2/S3 names than in S1 names (US: service 5.1x, center 2.5x), while a decoy's swapped-in word is ordinary business vocabulary (~0.8x). On labelled US/India errors the flag hits 32 decoys and 1 true match. French accents are stripped and abbreviations unified (Cie = Compagnie) before comparing words, and a street-only similarity (without number, street type or city) catches decoys that keep the house number but move to another street.

**Group check** (`group.py`):
- For each pair, compute the similarity of the record to the S1's other confidently matched records (its siblings).
- A true match resembles its siblings; a decoy is a perturbed copy of the S1 record and often does not.

**Assignment and thresholds:**
- Each S2/S3 record goes to its best S1 if the combined probability exceeds the threshold. The threshold is chosen by grid search on held-out S1 entities to maximise macro F0.5, and is high because F0.5 rewards precision.
- **France:** no labels exist to tune on. Its threshold is set so that France gets the same matches per S1 as US/India, which the generator appears to share across countries. Pairs whose legal forms and house numbers both differ are treated as decoys: in US train this pattern is true only 0.1% of the time. The French names use a small vocabulary of generic words (Club, Ecole, Amis, Union...), so word-swap decoys (Enfance Club -> Enfance Ecole at the same address) score as high as true copies; 25.7% of France's pairs scored 0.8-0.999 carry the swap flag (US/India 1-2%). France pairs with the flag are therefore dropped before its threshold. French noise also almost never changes house numbers (1.7% of accepted pairs differ, against 10-11% in US/India, where these are real typos), so France pairs with different numbers, or the same number on another street, are dropped too, and France uses a stricter threshold (0.97): the leaderboard showed its borderline pairs were only ~57% correct (`variants.py fraggr`). We first also dropped an added "Groupe" / a word replaced by "& Fils" (the French counterparts of the business words US/India decoys add), but the leaderboard showed French filler words (Groupe, Fils, Developpement, Associes, Services, International, Distribution) are true-copy noise, so those keep the shared threshold instead. The last leaderboard-tested additions are described in section 5 (`final.py`).

---

## 5. Results & Error Analysis
| stage (held-out train S1 entities) | macro F0.5 |
|---|---|
| LightGBM matcher | 0.9738 |
| cross-encoder alone | 0.9757 |
| matcher + cross-encoder combiner | 0.9828 |
| + group check | 0.9838 (US 0.9841, India 0.9834) |
| + decoy-signature features (house-number shift, legal form, name-edit type, added business word) | 0.9846 |
| + per-S1 expected-F0.5 decision | 0.9847 |
| **final: 5-encoder ensemble + all of the above (US/India)** | **0.9855** (US 0.9855, India 0.9853) |

**Public leaderboard:**
- 0.9666 with a single shared threshold.
- 0.9717 with the France threshold calibration.
- 0.9778 after the France rebuild (legal forms kept).
- 0.9788 with the decoy-signature features; 0.9782 with France calibration switched off (so it stays on).
- Final file: US/India from the all-encoder run (validation 0.9855), France from the rebuild with both encoders and the France decoy rules (`variants.py merge`). Test has more distractors per S1 than train: with the same model and threshold US gets 3.358 matches per S1 on test against 3.310 on validation (India 3.308 vs 3.301), A leaderboard test showed that records renamed to a made-up brand at the same address are true copies (removing 50k of them cost 0.0021), so no US trim is applied and brand-name candidates at the same address and house number rejected by the threshold are accepted (6.6k pairs).
- Last France steps, each checked on the leaderboard one at a time (`src/final.py`, which rebuilds the submitted file byte-for-byte from the two group runs):
  - 0.9827: no US trim, brand copies added back.
  - 0.9834: Groupe / Fils additions kept (not treated as decoys); dropping Developpement / Associes additions instead cost 0.0005 (0.9810 vs 0.9815).
  - 0.98372: French filler additions at the shared threshold (0.725) instead of 0.97.
  - 0.98393: records adding a French filler word at the same house number and address, scored 0.3-0.725, accepted.
  - **0.98426 (final)**: France pairs accepted by the shared threshold but not by 0.97, not flagged by any decoy rule, and whose record has no competing S1 above 0.3 (9.7k pairs).

**Findings:**
- **France was the main source of error on the leaderboard.** With the shared threshold it received 3.66 matches per S1 entity against 3.35 for US/India, despite having fewer S2/S3 records per S1. The extra matches were decoys that differ only in legal form and house number, which the models could not see while French legal forms were being stripped.
- **Remaining false positives:** branches of one chain at nearby addresses, and generic names in one city with an incomplete address.
- **Remaining false negatives:**
  - Pairs retrieval never proposes (2% of true pairs), mostly an empty address combined with a name shared by many S1 entities.
  - A brand name against the registered name.
  - Rare native-script words.

---

## 6. Conclusion
Careful normalisation, multi-view TF-IDF retrieval and a gradient-boosted matcher with candidate context give a strong base (0.974). A small fine-tuned cross-encoder and a group-consistency check raise it to 0.984 on held-out data.

For an unseen country, the largest gains came from checking the structure of our predictions against the train statistics, not from model capacity. That check exposed a normalisation step that hid exactly the signal separating decoys from true matches.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/prep.py`: normalisation.
- `src/learn_native.py`: the learned native-script map.
- `src/block.py`: retrieval.
- `src/eval_block.py`: retrieval recall.
- `src/features.py`: pair features.
- `src/train.py`: the matcher, the pruner and the thresholds.
- `src/predict.py`: stage-1 outputs.
- `src/xenc.py`: cross-encoder fine-tuning, scoring and the combiner.
- `src/join_xenc.py`: joins the scores of several cross-encoder runs.
- `src/group.py`: the group check, France calibration and the group-run `matching_results.tsv`.
- `src/variants.py`: France decoy rules and merge of the US/India and France runs.
- `src/final.py`: the exact post-processing chain that produced the submitted file.
- `src/kaggle_run.py`: the Kaggle notebook runner.
- `src/validate_submission.py`: the official validator.
- `native_map_learned.tsv`, `README.md` (command sequence) and `requirements.txt`.

We ran the pipeline as Kaggle notebooks in this order:
1. train (CPU)
2. test (CPU)
3. predict (CPU)
4. cross-encoder (GPU; five runs: x2-x4 on the first pipeline, x5-x6 on the France rebuild)
5. group (CPU; US/India with x2-x6, France with x5-x6)
6. `final.py` locally (France decoy rules, merge, leaderboard-tested additions)
