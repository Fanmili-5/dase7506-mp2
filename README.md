# DASE7506 MP2: Choosing among four mathematical solutions

This repository contains the model that answered **399/467 test questions correctly (85.4390%)**, its training code, complete checkpoint and [five-page report](report/MP2_Report.pdf). The editable report is [here](report/MP2_Report.md).

The method combines a linear verifier with two small neural verifiers. It uses the supplied candidate representations and arithmetic/number-source features. Inference runs on CPU with NumPy; it does not need an LLM or a network connection.

## Reproduce the score

The recorded environment is Linux, Python 3.12.15 and NumPy 2.5.3. From a Linux terminal:

```sh
git clone https://github.com/Fanmili-5/dase7506-mp2.git
cd dase7506-mp2
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/check_release.py --ready-to-submit

git clone https://github.com/xudongwu-0/DASE7506-MP2.git ../mp2-official
git -C ../mp2-official checkout bb801f4f2e7629d270df765ab12c333b2dc616bb

python scripts/evaluate_direct_selected_20261031.py \
  --snapshot inference --split test --final-test \
  --data-dir ../mp2-official/code/data --output runs/test_reproduction
```

The evaluation command runs the unchanged official predictor and scorer in separate fresh CPU processes. It disables CUDA, sets numerical libraries to one thread, restricts CPU affinity, and monitors the complete time and thread count. The predictor cannot open the reference-answer file. Use a new output directory for another run.

The expected result is `correct=399`, `n=467` and `accuracy=85.4389721627409`. The preserved official result is [evaluation/test-score.json](evaluation/test-score.json).

All ten checkpoint files are already in `inference/checkpoint/`. They can also be downloaded as a [complete ZIP](https://github.com/Fanmili-5/dase7506-mp2/releases/download/mp2-85.4390/MP2_checkpoint_85_4390_20261031.zip). If using the ZIP, unpack it into `inference/`, giving `inference/checkpoint/config.json`. It is the same checkpoint as the repository copy. No training is needed for evaluation.

## Reproduce training

With the virtual environment above active, install Torch and expose one GPU no stronger than an RTX 4090:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_extended_training.py \
  --data-dir ../mp2-official/code/data --output runs/training_reproduction
```

This runs six linear fits and twelve neural fits serially, then assembles the submitted configuration. It fits only supplied training data and uses validation for early stopping and selection. The replay workspace contains train/validation inputs only; it does not score test or replace the released checkpoint.

The command completed in 274.07 seconds on the recorded server. All ten checkpoint files and all 62 arrays matched the released version exactly. Details, including the earlier launcher error and its directory-only fix, are in [training_reproduction.json](evidence/training_reproduction.json). This establishes reproducibility in that environment; other hardware or library versions can change floating-point results.

## Read the implementation

| Start here | Purpose |
| --- | --- |
| [improvement_selected_verifier.py](inference/methods/improvement_selected_verifier.py) | Final score combination and `predict` entry point |
| [direct_selected_verifier.py](inference/methods/direct_selected_verifier.py) | Two-model combination and per-question normalization |
| [relative_verifier_selector.py](inference/methods/relative_verifier_selector.py) | Linear verifier and answer grouping |
| [joint_verifier_selector.py](inference/methods/joint_verifier_selector.py) | NumPy neural inference and preprocessing |
| [training/](training/) | Training routines and neural model definitions |

The report explains the losses, seed choices, ablation and limitations. CPU validation is 751/798 overall and 308/355 on disagreement questions. The saved GPU comparison gave 309/355; one close choice changed after CPU deployment. This difference was recorded before the final test and was not adjusted away.

## Resources and experiment records

| Item | Recorded result |
| --- | --- |
| Complete test prediction plus independent scoring | 3.136154 seconds, Xeon Platinum 8368Q |
| Inference assets, uncompressed | 11,808,974 bytes (11.26 MiB) |
| Observed total threads | 2, including the monitor |
| Successful training replay | 18 serial fits on one RTX 4090 |

The course limits are 30 seconds on the reference Xeon 8457C, 16 MiB of assets and four total threads. Local measurements satisfy these thresholds; reference-machine timing eligibility remains for the instructor to determine.

Original development ran pairs of independent experiments concurrently on two 4090 cards. Each fit used one card; the successful replay was serial on one card. The course's one-GPU rule has not been confirmed to permit the earlier concurrent schedule. The report and [self-review](docs/SELF_REVIEW_20261031.md) retain this point explicitly.

The recent family used 18 fits. Earlier search records are retained separately, with overlap noted rather than treated as an exact total. The [evidence/](evidence/) and [evaluation/](evaluation/) directories contain selection records, costs and official results. [CHECKPOINT_MANIFEST.json](CHECKPOINT_MANIFEST.json) lists the assets; [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) records the code and document hashes. The predictor does not read these records.

## Attribution

The relative-verifier design follows Anchmy's public implementation, independently reimplemented here. The network uses ideas from TabM and set attention. [AI_AND_REUSE.md](AI_AND_REUSE.md) and the report disclose sources and substantive Codex assistance. Keep [LICENSE.txt](LICENSE.txt) and [NOTICE.txt](NOTICE.txt) with the release.

The inference freeze remains `4ab3279f6f968f80733fa3cf35c0ee17674f93b40c86c0fd5894234677bfe3a9`. This documentation revision changes the explanation and report, not the model, checkpoint or official score. Supplied raw data and prediction caches are not redistributed.
