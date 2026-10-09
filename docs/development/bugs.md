# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，历史证据留在 Git 与对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

新增一条用下面的块，**一条一事**。

```text
## B-NNN 短标题
- 状态：open | investigating
- 看见：怎么复现、期望 vs 实际
- 落点：源文件 / Study / 测试
- 方案：
- 验证：
```

---

## 开放

## B-019 探针准备阶段拆分遗漏环境模块导入

- 状态：investigating
- 看见：Flow 基线 d73b217、新声明提交 5f1ff5c 下运行 main_probe / MNIST-linear（f6db09ee2f38fec9），CPU parity 通过后写 ENVIRONMENT.json 抛 NameError: os is not defined。期望 prepare 完整写出环境并进入 execute；实际 failed，尚未正式训练
- 落点：studies/main_probe/prepare/data.py，拆分时保留的函数使用 os.environ，但模块未导入 os
- 方案：补充导入，核对新模块的名称依赖；保留 version01 失败，以新 version 从头复验
- 验证：失败 result 与 logs/run.log 位于 studies/main_probe/runs/f6db09ee2f38fec9，原始准备证据保留在 assets/probe（本机产物，不随 Git clone 提供）；尚待修复后真实 GPU 全阶段复验

## B-020 探针来源清单包含自身生成的归档文件

- 状态：investigating
- 看见：同一失败 Run 的 SOURCE_MANIFEST.json 显示 before105 / after137 文件；新增32个条目均是本 Run 生成的 source-main-98648f3/src/*.py，库与 Study 源码没有实际编辑。数据与模型 CPU 数值门通过，但 prepared.source_unchanged=false，导致总体准备无法通过
- 落点：studies/main_probe/execute/paired.py 的 source_manifest 递归扫描整个 Study，错误包含忽略的 runs 产物
- 方案：复用库 provenance.source_files，只记录声明与源码；原归档仍逐字节对照 Git blob，并保留独立归档完整性检查，不改变数值门
- 验证：新增生成归档不改变库来源清单的合同测试，再用 version02 复验真实 GPU 全阶段；尚待完成

下次新增问题从 **B-021** 起编号。

已关闭项不在此处重复列出；修复、验证及维护关闭的不同口径见 [record.md](record.md)、[brainstorm.md](brainstorm.md) §4 和对应 Study 报告。没有开放条目不等于所有历史根因已查明或所有场景已验证。
