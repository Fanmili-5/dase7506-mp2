# AI与复用披露

OpenAI Codex实质参与实现、方法比较、GPU训练、验证选型、CPU部署核验、打包和文字。学生须理解代码并披露帮助。

相对特征/分块正则/答案聚合设计参考Anchmy/dase7506-mp2-group-relative-verifier固定08813bf6ad85fea5242fc40392f25dd8ba26e1cf，此处独立实现，未用其权重、数据、测试预测/逐题结果；其公开394/467总分在实验前已知。

网络借鉴TabM（ICLR2025，arxiv2410.24210）的共享矩阵/成员缩放/逐成员目标及Set Transformer（Lee等，ICML2019，PMLR97）的集合交互；自有小型适配，不冒称官方作者模型复现。模型只用课程train拟合，validation开发；没有额外数据、预训练权重、LLM生成或语义金标。官方common.py/predict.py/score_predictions.py保持原样。数据归属见保留LICENSE/NOTICE。
