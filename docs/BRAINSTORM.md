# Brainstorm

> 未拍板的想法。**不**当合同。权威是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → [STUDY_GUIDE.md](STUDY_GUIDE.md)。缺陷进 [BUGS.md](BUGS.md)。

**对照**的基准是 git **`main`**（旧 RPipe 行为：脚本、`process.py`、native 训练与 metric）。**借鉴** DeepScientist：durable 契约与编排纪律，不把它当对照物，也不做成研究 OS。

新想法追加在 **§3**。拍板后写入 CONCEPT / LAYOUT / STUDY_GUIDE，并从这里收口。

---

## 1. 对照 `main`

当前树要能对上 `main` 上已经成立的执行习惯，尤其是 native 这一支：

- 调度形状：`&` + `wait`，一组结束才开下一组
- Study 级 `process` 是 launch 之后的单独收口（对齐 `process.py`）
- metric 名、Accuracy 0–1、latest / best checkpoint、DataLoader shuffle / step 续训
- 对照 `main` **只钉** `custom_torch`；HF Trainer 等是并列 source，不要把 Trainer 特有键抬成 Control 必须表

开放：哪些 `main` 行为还没迁过来、要不要迁，写在 §3。

---

## 2. 借鉴 DeepScientist

学 durable 契约与编排纪律。定位仍是执行底座，不是 Quest / Canvas / Findings。

仍放在库外（除非 CONCEPT 改口）：daemon / Web / 决策器。

---

## 3. 开放想法

一条一事。先写清**已经有的**，避免把现成能力再当缺口。

### 这三件已经有了（先前 §3 写错了）

**process 有两层，Experiment 的统计挂在 Study 信封里。**

- 每个 Run 的 Flow 最后一阶段会写 `runs/<id>/process.json`（只这一次：metrics + history）。
- launch 全部 wait 完会再跑一次 **Study process**（也可单独 `rpipe process`），写根上 `process.json`：`scope: study` 信封。
- Experiment **没有**自己的 process 文件。跨 seed 的 **mean / std / min / max**（metrics 和曲线）写在信封的 `experiments[]` 里，图是 `docs/figures/learning_curves.png`。

所以不是「还没有 Study process」，也不是「只有 Run 层」。缺的如果还要想，是 vis 多不多、表够不够对上 `main` 的 `process.py`，不是统计公式本身。

**make / launch 的「接着跑」也有，但跟算法 checkpoint 不是同一个词。**

| 哪一层 | 现在做什么 |
|--------|------------|
| make / launch 跳过已成功 | 默认 **不**把已经 `succeeded` 的格子再排进 jobs。`launch` 有 `scripts/jobs.json` 就复用，只跑未成功的。`--include-done` 把清单里已成功的也排进去（不为此重做 make）。`--remake` 才重新 make。 |
| `--mode` | 仅这一次 launch 过滤 `train` / `eval`；**不**改写 `jobs.json`。 |
| launch 失败再试 | 各组 `wait` 完，对失败格子再跑一遍，日志写 `retry … (resume latest)`。再调的还是 `run-one`，**不改 yaml**。 |
| 算法 resume | train 缺省 `latest`（没文件就从头）；eval 缺省 `best`（sibling train）。这才是读 checkpoint。 |
| `split-round` | make CLI **已经有**这个开关。 |

没有的是 `main` 那种名叫 `resume_mode` 的 make 参数。不必为了对齐再造一个同名开关，除非现有三层不够用。

开放的只剩细处：retry 是否应写进 config 的 `resume: latest`；已 `succeeded` 但还想多训几个 epoch 怎么办（现在会被 skip 掉，除非 `--include-done` 或 wipe）。

### 对照 `main`：还可以想的（窄一些）

**Study process 的图/表是否够对照 `process.py`。** 统计和 mean±std 曲线已经有。`main` 还会按 control 出更多 png。要不要加图，另说；不要再立项「做聚合」。

**CIFAR 等小网格对照 study。** 数据和模型已注册，还没有第二条非 MNIST 对照。

**`mode=inference`。** 相对 `main` 是扩展，不是对照义务。

**DDP / TensorBoard / vLLM / Kornia。** 不迁。

### 借鉴 DS（仍不是对照）

只借「写下的东西还能被指认」。不做 Quest / Canvas / daemon / 决策器。

还可能想、且和现有能力不重复的：index 只读列出（CLI，非 Web）；失败草稿不覆盖已成功 `result`（原则已有，盯实现）；Study 记录版本/数据 digest。

### 下一刀（未拍板）

1. 不重做 mean/std。若还要对 `main` 的 `process.py`，只问还要不要更多图/表。
2. launch 的 skip / retry 已经能接着跑；开放的是「已成功还想再训」和 retry 要不要写进 yaml。
3. CIFAR 小网格当第二对照 study。
4. inference / 分布式 / 服务化往后推。
