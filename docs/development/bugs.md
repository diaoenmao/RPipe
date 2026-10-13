# Bugs

> 这里只保留尚未解决的缺陷。编号只增不复用；修复并验证后移除条目，开发证据留在 [record.md](record.md)，实验依据留在对应 Study 报告。正常差异或决定不修的事项不列为开放缺陷。权威仍是 [concept.md](../code/concept.md) → [layout.md](../code/layout.md) → 代码。

更新日期：2026-10-14。

## 开放

### B-054：Issue #36 在关闭 Study 扩展时仍覆写成功（investigating）

- 复现：freeze=true、flow.study_phases=false 的新 CPU Toy/Stub 两步 Run 成功后改 study.yaml 注释，原生run-one因来源变化exit1，却把原成功result改成failed；同一测试的扩展开启对照通过。
- 期望：prepare中的初始来源拒绝也应保留既有成功字节；真正的prepare / execute / summarize / write异常仍写failed。
- 落点：FlowRunner当前只在current=None的冻结异常保全；未扩展的冻结检查在prepare初始段触发。方案只扩展该异常的保全范围，不改变冻结启用条件、时机或训练行为。
- 红测：`.tmp/shared-inputs-20261014/issue36-hooks-red`，1 failed / 10 passed；原stdout、stderr与成功副本保留，不随clone提供。本地修复后定向32 passed、core456 passed / 2 skipped、CPU integration68 passed；等待PR最终CI / 集成，#36仍open，见 [修复记录](record.md#37-issue-36-未开启扩展的初始冻结拒绝b-054)。

### B-053：data / model 只在单个 Study 内共享（investigating）

- 复现：`artifact_layout(project/studies/a, id)` 与 `artifact_layout(project/studies/b, id)` 返回不同的 shared data / model 根；相同下载在不同 Study 重复存储。main_probe.prepare 还为原版 / 当前版逐 Run copyfile 完整 raw。
- 期望：项目内统一输入根；同版本 raw / 预训练权重一份，Run 状态独立；无法共用的派生格式显式私有。
- 落点：artifact layout、data prepare / profile、Flow prepare、Study 手动准备和 probe raw 适配；方案见 [目录合同](../code/layout.md#跨-study-共享资产2026-10-14)。
- 当前：默认公共根、私有例外及来源分版processed / 固定Stats均本地实现；等待工作PR最终CI / 集成。共享专项26 passed、core463 passed / 2 skipped、CPU integration68 passed、两个安装包各9CLI / 4Toy Run通过；完整raw hash与旧来源保持。依据见 [最新记录](record.md#38-固定格式-processed--stats-也跨-run-复用)。旧实测不是改动后的完整Study验收。

本轮按用户要求只保留本机适配及 Issue #36 / #37 / #38 的本机修复，见[当前范围与验证](record.md#32-按用户要求收缩到本机适配与三个下游-issue)。范围外人工反例的代码改动已撤回，不据此宣称这些问题不存在或当前已修复；历史事实留在开发记录，未升级为本轮待实施队列。B-030已补充report来源校验并完成定向 / CPU回归；原复现、修复与验证边界见[开发记录](record.md#9-b-030-报告来源校验与分支清理)。换version / fresh make后须重新process，不能导出身份不匹配的旧聚合。没有开放条目不等于所有后端、设备或研究条件均已实测。

B-002、B-021 / B-024及B-025至B-029的既有修复证据仍在 [record.md](record.md)。B-022 / B-023 的代表性容量、精度合同与探索边界保留各自报告；本轮新的设备 / 并发结果见 [本轮执行记录](../../studies/main_exp/docs/v030_execution.md)。未实测支持范围不是已通过，缺环境依赖的导入失败也不自动列为库缺陷。

B-031 至 B-054 已使用，编号不复用。下次新增问题从 **B-055** 起编号。

新增条目一条一事，记录编号、open / investigating状态、复现条件、期望 / 实际、影响落点、方案和实际验证证据。
