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

一条一事。下一轮对照是 `studies/main_base`：一个 Study，先跑 4 step 探针，再用实测 step 时间展开 `main` 的 60 step 网格。两种规模不要同时放进 `axes`，`launch` 不能按规模过滤。B-007 留在 [BUGS.md](BUGS.md)，不在这里重做。

**先不做：** 数据/库指纹；已成功还想多训几个 epoch；process 再多几张图。

**明确不做：** 把 `inference` 当对照义务；DDP；vLLM；TensorBoard。Kornia 只用于模型入口：Normalize，以及训练态的 flip / crop。

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
| `rpipe status` | 只读，实现在 `structure/artifact/readout/`。`pending` 的 `note` 是最后一条 `[epoch]` 或 `[error]`；`failed` 的 `note` 是最后一条 `[error]` 摘要，`error` 列仍是 result。`succeeded` 的 `note` 是 `-` |
| launch 阶段 | 每组开始打 `launch: wait i/n mode=`；再试打 `launch: retry`。结束打 `planned` / `succeeded` / `failed` / `pending`。不改 `jobs.json` |
| `rpipe logs` | 只读，同一份 readout。各 Run 事件行按时间打到终端。不写 Study 级总 log |
| `rpipe report` | 同一份 readout。从 `process.json` 写 `docs/NUMBERS.md`：Experiment mean / std / min / max 和 Run 表。不改 `STUDY_REPORT.md` 的结论 |
| `split-round` | make 已有 |
| CIFAR 小网格 | `studies/cifar_grid/`。2026-09-30 从头重跑，8/8 succeeded，没有 `torchvision::nms`。结论在该 Study 的 `docs/STUDY_REPORT.md` |
| `run.log` 行格式 | `时间 级别 Run id [事件] 内容`。时间是 RFC 3339 毫秒+时区。事件：`[flow]` `[error]` `[warn]` `[epoch]` `[split]` `[metric]` `[time]` `[ckpt]` `[resume]`。traceback 每一行都是 `[error]`。不改 `result.json`。PR #8，`e9a2f6b`，已进 `dev` |
| 测试规范 | `docs/TESTING.md` 采用 2026-09-28 正式规范。用例声明 `cost_class` 和 `result_type`。PR #8 当时 `--core` 142、`--all` 151。其后的 status / logs / report 用例还在本地，通过数以当次 `tests/run.py` 为准 |

不造 `main` 的 `resume_mode` 同名开关。
