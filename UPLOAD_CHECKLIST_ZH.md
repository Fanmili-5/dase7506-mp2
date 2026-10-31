# 上传步骤：85.4390%版本

本版本官方成绩 **399/467，accuracy = 85.4389721627409**。截止时间是 **2026年10月31日23:59（香港/北京时间）**。材料已在本地整理，未替你发布或提交。

## 1. 检查本版本材料

- 报告：`report/MP2_Report.pdf`，方法和成绩已更新。
- 官方评分：`evaluation/test-score.json`，网站准确率按其中`accuracy`填写，单位为百分比。
- 代码和精确命令：`README.md`、`inference/`、`scripts/`、`training/`。
- 全部权重、配置和统计：`inference/checkpoint/`，11,808,974字节未压缩资产。
- 实验/种子/资源/复现证据及AI披露：`evidence/`、`evaluation/`、`AI_AND_REUSE.md`。
- 完整性检查：`python scripts/check_release.py --ready-to-submit`。

CPU实际验证308/355与GPU缓存309/355有区别；这些都不是表单分数。网站应填 **85.4389721627409**。

## 2. 由你上传公开GitHub仓库

把`mp2-submission`目录内的内容放在公开仓库根目录，保留子目录层级。不要混入原始数据、`.venv/`、重训`runs/`或其他旧模型包。报告PDF通过代码仓库提供。

一种直接做法：先在GitHub新建一个**公开、空的仓库**，例如`dase7506-mp2`（不勾选自动创建README），然后在Mac终端执行。最后两行中的地址改成你刚建的仓库地址；这些命令由你执行，本包没有替你上传。

```sh
cd "/Users/fanmili/Documents/ChatGPT/hkuwork/output/MP2_final_submission_20261031/mp2-submission"
python3 scripts/check_release.py --ready-to-submit
git init
git add .
git commit -m "MP2 frozen submission 85.4390 percent"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/dase7506-mp2.git
git push -u origin main
git rev-parse HEAD
```

只在这个新的`mp2-submission`目录执行，不要把整个`hkuwork`工作区上传。推送后从GitHub网页确认报告及`inference/checkpoint/`的10个文件均存在。若使用其他Git客户端，上传同一目录的全部内容即可。

提交代码形成固定commit后，Code链接建议指向该commit对应的`tree`页面。版本标识为`mp2-85.4390-4ab3279f`，完整模型freeze SHA见`SOURCE_VERSION.json`。

Checkpoint可以随完整仓库保留在`inference/checkpoint/`，也可把单独checkpoint ZIP上传为GitHub Release资产。所填链接应能下载全部配置、权重和统计；仅一个神经权重文件不完整。独立checkpoint ZIP解压到`inference/`，应得到`inference/checkpoint/config.json`。

最省步骤是直接使用仓库中的完整权重目录：Code填`https://github.com/你的账号/仓库/tree/刚才的commit`，Checkpoint填同一commit下的`inference/checkpoint/`目录链接。`SUBMISSION_FIELDS.json`只是一份待填写参考，不必修改包内已核验文件；学号和公开链接在官方网站填写。

## 3. 官方表单与GitHub Issue

打开：https://xudongwu-0.github.io/courses/dase7506/mp2/

1. Student ID填写你自己的学号。
2. Accuracy填写`85.4389721627409`。
3. Code链接填写公开代码仓库固定版本，仓库中包含报告和证据。
4. Checkpoint链接填写完整权重目录固定链接，或Release下载链接。
5. 网站生成Issue链接后，点击 **Submit on GitHub**，并在GitHub上真正创建Issue；仅生成链接不算提交。
6. 确认Issue已创建，学号、分数和两个公开链接都正确。后续更新使用同一GitHub账号和学号。

最终条目必须在截止前包含代码和checkpoint/configuration链接。报告最多10页，通过代码仓库提供。正式成绩和资源资格由复现及教师审核决定。

## 4. 阅读后提交

确认你理解实现并保留实质AI帮助及参考方法披露。报告如实记录了历史测试总分暴露、CPU/GPU验证差异、开发并发和串行单卡重放，不要改成未经验证的“独立测试”或“任意机器逐位重训一致”。
