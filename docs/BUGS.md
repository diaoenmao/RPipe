# Bugs / 待办

> 库内已知缺陷与跟进项。不是路线图：OS / Web / 决策器见 [CONCEPT.md](CONCEPT.md)。权威仍是 CONCEPT → LAYOUT → 代码。

新增一条用下面的块，**一条一事**。修完把状态改成 `done` 并写 PR / commit；不要删历史（短因即可）。

```text
## B-NNN 短标题
- 状态：open | investigating | wontfix | done
- 看见：怎么复现、期望 vs 实际
- 落点：源文件 / Study / 测试
- 下一步：
```

---

## B-001 Windows OpenMP 双份运行时，make/launch 直接退出

- 状态：open
- 看见：本机 conda + torch 下 `python -m rpipe make …` 报 `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`，进程退出码非 0。设 `KMP_DUPLICATE_LIB_OK=TRUE` 后能跑完。
- 落点：Windows 运行时 / 启动入口（`flow/cli.py` 或文档）。不是算法对不齐。
- 下一步：确认是否只在 Anaconda 混装 OpenMP 时出现。若常见，在 cli 启动时给出可读错误，或 STUDY_GUIDE 写清环境约束。不要把 `KMP_DUPLICATE_LIB_OK` 默认写进库里当正式修复。

## B-002 `system.device: cpu` 时 make 仍按 GPU 装箱并打印 GPU 名

- 状态：open
- 看见：`studies/mnist_native_vs_hf` 基底是 `device: cpu`，`make --num-gpus 1 --init-gpu 0` 仍打印 `GPU0 NVIDIA … usable …GiB` 和 wait 组。launch 日志也是 `+ gpu=0 <run_id>`。
- 落点：`structure/make` 排班 / `flow/cli` 启动标签
- 下一步：cpu Run 应走 CPU 队列（或明确「只借用 GPU 槽位做进程数」）。至少 pack 文案不要暗示这些 job 占着那张卡的显存。

## B-003 HF Trainer 与 native 在 train_size=8000 上 last accuracy 差约 0.0008

- 状态：wontfix（记录，除非差变大）
- 看见：`studies/mnist_native_vs_hf`（2026-09-16）。500 / 2000 三颗 seed 的 accuracy、best_accuracy、train_loss **逐点相同**。8000：HF last accuracy mean 0.9069 vs native 0.9061（Δ ≈ +0.0008），与 seed 标准差同量级。
- 落点：`structure/algorithm` 的 `transformers_trainer` cosine / `_EpochScheduler`
- 下一步：不改 native 去追这个数（见该 Study PLAN）。若 500/2000 开始对不齐，再当作回归打开。

## B-004 单测里 `lr_scheduler.step()` 早于 `optimizer.step()` 的 PyTorch 警告

- 状态：open
- 看见：`python tests/run.py --core` 对 `test_optim.py` / `test_train.py` 打 `UserWarning`。用例仍通过。
- 落点：`tests/rpipe/structure/algorithm/test_optim.py`、`test_train.py`（以及若生产路径同样顺序，则 `algorithm/optim.py`）
- 下一步：先看测试是否只为了读 lr 而单独 `sched.step()`。生产循环若也反了，按 PyTorch 1.1+ 约定改顺序。
