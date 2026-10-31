# Sources and attribution

The relative features, block regularization and answer-group scoring follow the design in [Anchmy/dase7506-mp2-group-relative-verifier](https://github.com/Anchmy/dase7506-mp2-group-relative-verifier/tree/08813bf6ad85fea5242fc40392f25dd8ba26e1cf), independently implemented here. No weights, data or individual test predictions were imported. The repository's published aggregate score was visible when the design was studied.

The neural models adapt parameter sharing and member scaling from [TabM](https://arxiv.org/abs/2410.24210) and candidate interaction through set attention from [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html). All model parameters are trained from scratch on supplied course data.

The official parser, predictor and scorer are retained unchanged. GSM8K and Qwen attribution is preserved in LICENSE.txt and NOTICE.txt. The report contains the assistance acknowledgement.
