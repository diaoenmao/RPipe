# Brainstorm

> 未拍板的想法。**不**当合同。权威是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [STUDY_GUIDE.md](STUDY_GUIDE.md)。缺陷进 [BUGS.md](BUGS.md)。

**对照** git **`main`** 的执行形状。**借鉴** DeepScientist 的账本纪律，不当对照物，不做研究 OS。

前面只写规则和还要做的。已经落地的在文末，不占对照清单。

新想法追加在 **§3**。拍板后写入 CONCEPT / LAYOUT / STUDY_GUIDE，并从这里删掉。

---

## 1. 对照 `main`（硬性）

对照的是形状：调度、Study 收口、metric / checkpoint 习惯。

**不对照：**「只做 `custom_torch`」。旧 `main` 只有这一支；这边 Registry 并列挂多个 `source`。Trainer 特有键不进 Control 必须表。

---

## 2. 借鉴 DeepScientist（硬性）

只借：写下的东西还能被指认（index、result、status）。

**不做：** Quest、Canvas、Findings、daemon、Web、决策器。

---

## 3. 要做的

一条一事。

**`run.log` 用 `[tag]` 排版。** 现在一行是 `id` 后面接一长串空格分开的词（`epoch 5 test Accuracy 0.3415 Loss …`），失败、checkpoint、flow 起止混在同一种句子里。改成固定前缀加方括号标签，人眼和 `rg` 都能切段。同一份 `run.log`，行首仍是 Run `id`。不改 `result.json` 的字段。

示意：

```text
f487ffe68a52f01c [flow] start phases=prepare,execute,collect,summarize,write,process
f487ffe68a52f01c [error] phase=prepare RuntimeError: operator torchvision::nms does not exist
f487ffe68a52f01c [epoch] 5 [split] test [metric] Accuracy=0.3415 Loss=1.8424 [resume] best step=80
f487ffe68a52f01c [ckpt] best path=runs/…/checkpoints/best.pt
f487ffe68a52f01c [flow] succeeded
```

标签先只覆盖已经在打的几类：`[flow]` `[error]` `[epoch]` `[split]` `[metric]` `[ckpt]` `[resume]`。traceback 行保持 `[error]` 续行，不另造格式。`--console shared` 时仍靠行首 `id` 把多条 Run 拆开。

**先不做：** 数据/库指纹；已成功还想多训几个 epoch；process 再多几张图。

**明确不做：** 把 `inference` 当对照义务；DDP；vLLM；TensorBoard；Kornia 当库能力。

---

## 4. 已经做的

| 能力 | 口径 |
|------|------|
| `&` / `wait` | 一组结束才开下一组。`launch` 和 make 脚本 |
| Study process | launch 全部 wait 完后再收口；也可 `rpipe process`。信封在根 `process.json`；跨 seed 的 mean / std / min / max 在 `experiments[]`。Experiment 无文件夹 |
| metric / checkpoint | 名、Accuracy 0–1、latest / best、shuffle / 续训。native 已钉；其它 `source` 同一套键自己映射 |
| `--mode train` / `eval` | 只滤这一次 launch，不改 `jobs.json` |
| skip / `--include-done` / `--remake` | 已成功默认跳过；`--include-done` 复用清单；`--remake` 才再 make |
| 失败再试 | 组末 `retry … (resume latest)`，不改 yaml |
| 算法 resume | train `latest`；eval `best`（sibling） |
| 失败 log | 同一份 `run.log` 加 traceback；`result` 有 `status` / `error` |
| config 身份 | Run `id` 就是 config hash。`index.id` 是整张清单。不再做第二套 |
| `rpipe status` | 只读。见 STUDY_GUIDE §7。已进 `dev` |
| `split-round` | make 已有 |
| CIFAR 小网格 | `studies/cifar_grid/`。2026-09-26 launch 完，8/8 succeeded。报告在该 Study 的 `docs/STUDY_REPORT.md` |

不造 `main` 的 `resume_mode` 同名开关。
