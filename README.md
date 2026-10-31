# DASE7506 MP2: Relative and Joint Verifiers

**Official test: 399/467 = 85.4389721627409%.** Submit the accuracy field from `evaluation/test-score.json`. Frozen CPU validation is 751/798, including 308/355 disagreement questions. The report is `report/MP2_Report.pdf`.

Release ID: **mp2-85.4390-4ab3279f**. Inference freeze: `4ab3279f6f968f80733fa3cf35c0ee17674f93b40c86c0fd5894234677bfe3a9`. Original inference source and weights under `inference/` are unchanged. 中文上传步骤见 `UPLOAD_CHECKLIST_ZH.md`。This package does not publish or submit an entry.

## Install and obtain the data

Use Linux and Python 3.12 (observed: 3.12.15). CPU inference needs NumPy only.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/check_release.py --ready-to-submit

git clone https://github.com/xudongwu-0/DASE7506-MP2.git ../mp2-official
git -C ../mp2-official checkout bb801f4f2e7629d270df765ab12c333b2dc616bb
```

No banks, references, embeddings or prediction caches are redistributed. The official IO/parser/scorer are unchanged and their hashes are checked against the original release.

## Reproduce the frozen score without training

All assets are in `inference/checkpoint/`. If using the separate checkpoint ZIP, unzip it into `inference/` so `inference/checkpoint/config.json` exists.

```sh
python scripts/evaluate_direct_selected_20261031.py \
  --snapshot inference --split test --final-test \
  --data-dir ../mp2-official/code/data --output runs/test_reproduction
```

The command verifies the freeze and runs the original predictor and independent scorer in two fresh CPU processes. It hides CUDA, limits threads and affinity, measures startup/loading/preprocessing/all branches, and preserves existing outputs. Prediction cannot read references. Use a new output directory for another reproduction.

For direct official CLI reproduction, apply the same limits and time both processes:

```sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1
python inference/code/predict.py \
  --method inference/methods/improvement_selected_verifier.py \
  --checkpoint inference/checkpoint --split test \
  --data-dir ../mp2-official/code/data --output runs/test-predictions.json
python inference/code/score_predictions.py --split test \
  --data-dir ../mp2-official/code/data \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

The preserved complete fresh-process measurement is **3.136154 seconds**, **11,808,974 uncompressed asset bytes**, with two observed threads including the monitor, on Xeon 8368Q. Limits: 30 seconds, 16 MiB, four total threads; reference Xeon 8457C determines eligibility. Use the monitored entry for the recorded resource check.

## Reproduce training

Training uses supplied train labels; validation supports early stopping and selection. There are no external data/weights or test labels in fitting.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_extended_training.py \
  --data-dir ../mp2-official/code/data --output runs/training_reproduction
```

Expose one GPU no more powerful than an RTX 4090. The entry copies train/validation inputs into a new workspace, serially replays six logistic plus twelve neural fits, then assembles the already chosen configuration. It never scores test or replaces released weights. The submission launcher fixes the missing runs-parent directory; the original failure occurred before its first fit. A clean single-GPU replay completed in **274.071774 seconds**: all **10 checkpoint files were byte-identical**, and all **62 arrays matched exactly**, on the recorded server and dependencies. This is same-environment evidence, not a guarantee across hardware. Results are in `evidence/training_reproduction.json`.

Original development sources are in `training/`. Only host/UUID guards are made portable in the replay workspace; numerical recipes, seeds, preprocessing, losses and early stopping stay unchanged. It reconstructs this submitted family/configuration, not the earlier Gaussian/path project. Fresh weights do not automatically inherit a score; the attached original frozen checkpoint is the authority for 399/467.

## Method and evidence

The relative logistic verifier uses 1,024 relative embedding columns plus 32 relative arithmetic/answer-source features, block L2 penalties .05/.003, and answer-group max plus .5 log(count). Neural comparators use train-only standardization/ZCA, absolute and relative views, eight shared-weight members, and one four-candidate attention block. Seeds 17/29/43 were compared. The submitted mass branch is seed 17; pairwise branch is seed 29.

```text
S_previous = 0.5 Z(relative) + 0.5 Z(mass_joint_seed17)
S_final    = 0.5 Z(S_previous) + 0.5 Z(pairwise_joint_seed29)
```

Z standardizes the current question's valid candidate scores. The final comparator was nominated from existing validation-tied options and frozen before testing. GPU-cached validation was 309/355; CPU deployment is 308/355 because a combined-score difference below 1e-6 changed one choice. It was recorded before testing; weights, inference source and fusion coefficients were not adjusted. Earlier test totals were known; the report does not claim a pristine single-shot holdout study.

Eighteen fits describe recent family development, with clean replay costs recorded separately. Earlier exploratory costs are historical and non-additive. Development had two concurrent independent single-4090 jobs, never multi-GPU fitting; clean replay is serial on one GPU. Both schedules are disclosed.

`CHECKPOINT_MANIFEST.json` lists every inference asset. `RELEASE_MANIFEST.json` covers code/report/evidence hashes. Evidence is not read by prediction. Preserve `AI_AND_REUSE.md`, `LICENSE.txt`, `NOTICE.txt`, and publish the matching code version/report/complete checkpoint. The student publishes links and creates the official GitHub submission issue.
