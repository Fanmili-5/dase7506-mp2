# Submission checklist

Requirements: [official MP2 guide](https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md), checked on 31 October 2026.

| Requirement | Included material |
| --- | --- |
| Official complete-test accuracy in percent | evaluation/test-score.json: 399/467, accuracy 85.4389721627409 |
| Report of at most ten pages | report/MP2_Report.pdf and editable report/MP2_Report.tex |
| Research question, comparisons and ablation | Report sections on method, candidate models, objectives and component combinations |
| Resource measurements | evaluation/test-resources.json: 3.136154 seconds, 11,808,974 asset bytes, two observed total threads |
| Training and search records | Training sources, seeds 17/29/43, recent development costs and separate replay/history records |
| Training resources | Each training run uses one RTX 4090; independent experiments run on separate cards |
| Train-only fitting | Supplied training inputs and references; validation for development and early stopping |
| Complete inference assets | All ten files under inference/checkpoint, including learned transforms and configurations |
| Reproduction without training | Exact installation and evaluation commands in README.md |
| Original evaluation tools | Official common.py, predict.py and score_predictions.py, checked by hash |
| Attribution and acknowledgement | ATTRIBUTION.md, report references and one acknowledgement sentence in the report |
| Matching code and checkpoint | SOURCE_VERSION.json, CHECKPOINT_MANIFEST.json and RELEASE_MANIFEST.json |
| Public links and score submission | Public repository and checkpoint Release; the student completes the official form and creates the GitHub issue |

The local timing record uses a Xeon Platinum 8368Q; the course reference CPU is the Xeon Platinum 8457C. Full validation accuracy, disagreement validation accuracy and complete-test accuracy are identified separately in the report.

The final form uses the official JSON accuracy value. The final entry includes the fixed code version and complete checkpoint download link before 31 October 2026, 23:59 UTC+8.
