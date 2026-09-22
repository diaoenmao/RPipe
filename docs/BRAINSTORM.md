# Brainstorm

> 未拍板的想法。**不**当合同。权威是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [STUDY_GUIDE.md](STUDY_GUIDE.md)。缺陷进 [BUGS.md](BUGS.md)。

**对照** git **`main`**：旧 RPipe 的**执行形状**（`&`/`wait`、`process.py` 收口、metric / checkpoint 习惯）。不是「只做一个 backend」——旧栈只有 custom torch；这边要多个 `source`。

**借鉴** DeepScientist：durable 契约与编排纪律。不当对照物，不做研究 OS（daemon / Web / 决策器仍在库外）。

新想法追加在 **§3**。拍板后写入 CONCEPT / LAYOUT / STUDY_GUIDE，并从这里收口。

---

## 1. 对照 `main`

要对上、且已经落地：

- **调度：** `&` + `wait`，一组结束才开下一组（`launch` 与 make 写出的脚本）
- **Study process：** 整轮 launch wait 完后再收口（也可 `rpipe process`）。对齐旧 `process.py` 的位置：读结果做聚合/图，不是再训。信封在根 `process.json`；跨 seed 的 mean / std / min / max 在 `experiments[]`（Experiment 无文件夹）；Run 另有自己的 `process.json`

metric 名、Accuracy 0–1、latest / best、shuffle / 续训：native 已钉；其它 `source` 走同一套键，自己映射。不要把 Trainer 特有键抬成 Control 必须表。

**不对照：** 只做 `custom_torch`。Registry 并列挂 backend。

---

## 2. 借鉴 DeepScientist

学「写下的东西还能被指认」。定位仍是执行底座，不是 Quest / Canvas / Findings。

---

## 3. 开放想法

一条一事。已有的能力不重复立项。

### 已有（不要再做）

| 能力 | 现在 |
|------|------|
| `&` / `wait` | `launch` + 脚本 |
| Study process | launch 之后；`rpipe process` |
| `--mode train` / `eval` | 只过滤这一次 launch，**不**改 `jobs.json` |
| skip / `--include-done` / `--remake` | 已成功默认 skip；`--include-done` 复用清单；`--remake` 才再 make |
| 失败再试 | 各组完后 `retry … (resume latest)`，不改 yaml |
| 算法 resume | train `latest`；eval `best`（sibling） |
| 失败 log | 同份 `run.log` + traceback；`result` 里 `status` / `error` |
| config 身份 | Run `id` = config hash。报告表格已抄。`index.id` = 整张清单。不必再 hash 一遍 config |
| `rpipe status` | 只读：index + 各条 result。见 STUDY_GUIDE §7 |
| `split-round` | make CLI 已有 |

没有 `main` 那种 `resume_mode` 开关。不必为对齐再造。retry 要不要写进 yaml、已成功还想多训几个 epoch：用得少先不动。

### 可以想（未拍板）

**CIFAR 小网格对照 study。** 数据和骨干已注册，现有 study 都是 MNIST。对照旧 `make.py` 网格，不是新算法。integration，不进 `--core`。

**process 多图。** mean±std 曲线已有。只有对照旧 `process.py` 觉得缺图才做，不做第二套聚合。

**数据/库指纹。** 不是环境变量，也不是 config hash。是 rpipe 版本 + `shared/data` 文件哈希（数据或库变了 `run_id` 可以不变）。未拍板。报告钉整轮：抄 `index.id`。

**已成功还想多训。** 现在只有整格 `--include-done`。

### 明确不做

`inference` 当对照义务；DDP；vLLM；TensorBoard；Kornia 当库能力；daemon / Web / 决策器。

### 建议下一刀

1. CIFAR 小网格对照。
2. 其余往后。
