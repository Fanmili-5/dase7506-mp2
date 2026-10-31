# DASE7506 MP2: Choosing among four mathematical solutions

This repository contains a model for selecting among four mathematical solutions. It answered **399/467 test questions correctly (85.4390%)**. Training code, the complete checkpoint, the [report](report/MP2_Report.pdf) and its editable [LaTeX source](report/MP2_Report.tex) are included.

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

COURSE_DATA="/absolute/path/to/provided/code/data"

python scripts/evaluate_direct_selected_20261031.py \
  --snapshot inference --split test --final-test \
  --data-dir "$COURSE_DATA" --output runs/test_reproduction
```

Set `COURSE_DATA` to the `code/data` directory in the supplied course package. The evaluation command runs the unchanged official predictor and scorer in separate fresh CPU processes. It disables CUDA, sets numerical libraries to one thread, restricts CPU affinity, and monitors the complete time and thread count. The predictor cannot open the reference-answer file. Use a new output directory for another run.

The expected result is `correct=399`, `n=467` and `accuracy=85.4389721627409`. The preserved official result is [evaluation/test-score.json](evaluation/test-score.json).

All ten checkpoint files are already in `inference/checkpoint/`. They can also be downloaded as a [complete ZIP](https://github.com/Fanmili-5/dase7506-mp2/releases/download/mp2-85.4390/MP2_checkpoint_85_4390_20261031.zip). If using the ZIP, unpack it into `inference/`, giving `inference/checkpoint/config.json`. It is the same checkpoint as the repository copy. No training is needed for evaluation.

## Reproduce training

With the virtual environment above active, install Torch and expose one GPU no stronger than an RTX 4090:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_extended_training.py \
  --data-dir "$COURSE_DATA" --output runs/training_reproduction
```

This runs six linear fits and twelve neural fits serially, then assembles the submitted configuration. It fits only supplied training data and uses validation for early stopping and selection. The replay workspace contains train/validation inputs only; it does not score test or replace the released checkpoint.

The command completed in 274.07 seconds on the recorded server. All ten checkpoint files and all 62 arrays matched the released version exactly. The environment and replay records are in [training_reproduction.json](evidence/training_reproduction.json).

## Read the implementation

| Start here | Purpose |
| --- | --- |
| [improvement_selected_verifier.py](inference/methods/improvement_selected_verifier.py) | Final score combination and `predict` entry point |
| [direct_selected_verifier.py](inference/methods/direct_selected_verifier.py) | Two-model combination and per-question normalization |
| [relative_verifier_selector.py](inference/methods/relative_verifier_selector.py) | Linear verifier and answer grouping |
| [joint_verifier_selector.py](inference/methods/joint_verifier_selector.py) | NumPy neural inference and preprocessing |
| [training/](training/) | Training routines and neural model definitions |

The report explains the losses, seed choices and component comparisons. CPU validation is 751/798 overall and 308/355 on disagreement questions. Saved GPU development scores give 309/355; one close choice changes in CPU inference. The CPU result is reported for deployment.

## Resources and experiment records

| Item | Recorded result |
| --- | --- |
| Complete test prediction plus independent scoring | 3.136154 seconds, Xeon Platinum 8368Q |
| Inference assets, uncompressed | 11,808,974 bytes (11.26 MiB) |
| Observed total threads | 2, including the monitor |
| Successful training replay | 18 serial fits on one RTX 4090 |

The course limits are 30 seconds on the reference Xeon 8457C, 16 MiB of assets and four total threads. The table records local Xeon 8368Q measurements; official reproduction uses the reference CPU.

Every training run used a single RTX 4090. Development ran two independent experiments at a time, one on each card, with no multi-GPU training within either experiment. The successful replay runs all fits serially on one card.

The recent model family used 18 fits. Earlier search costs are recorded separately because some historical entries overlap. The [evidence/](evidence/) and [evaluation/](evaluation/) directories contain selection records, costs and official results. [CHECKPOINT_MANIFEST.json](CHECKPOINT_MANIFEST.json) lists the assets; [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) records the code and document hashes.

## Build the LaTeX report

The report uses standard LaTeX packages and includes its bibliography in the same source file. Compile it with Tectonic:

```sh
tectonic --outdir report report/MP2_Report.tex
```

It can also be compiled with pdfLaTeX on Overleaf. The report PDF is already included, so LaTeX is not needed for model evaluation.

## Method references

Candidate-context comparison relates to [groupwise ranking](https://arxiv.org/abs/1811.04415), L2 coefficient shrinkage to [ridge regularization](https://doi.org/10.1080/00401706.1970.10488634), and final-answer consensus to [self-consistency](https://arxiv.org/abs/2203.11171). The neural models use ideas from [TabM](https://arxiv.org/abs/2410.24210) and [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html). The report specifies the feature definitions, separate penalties and max-plus-log-count aggregation rule.
