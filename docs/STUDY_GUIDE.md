# Study 使用指南

如何用 RPipe 做一轮可复现实验。权威概念见 [CONCEPT.md](CONCEPT.md)；本页是**操作契约 + 模版**。

---

## 1. 推荐流程

1. 选 / 写 Experiment（Structure + Flow 能力；`experiment_config.yaml` 基底）  
2. 新建 Study 目录，**先填 `study.yaml`**（变量轴、tags、描述）  
3. 根据 `study.yaml` 展开 → 写各 Run Config → 写 **`index.json`**（launch 前）  
4. launch Flow：`prepare → execute → collect → summarize → persist → process`  
5. 读 `artifact/runs/<id>/result.json`；需要时写 `RESULTS.md`

不要跳过第 2 步直接手写一堆 patch。

---

## 2. 目录（目标布局）

```text
examples/studies/<study>/
  study.yaml
  index.json
  PLAN.md / RESULTS.md          # 可选人文记录
  artifact/
    shared/data/
    shared/model/
    runs/<run_id>/{config.yaml,result.json,assets/}

examples/experiments/<experiment>/
  experiment_config.yaml        # 基底
  grid/  launch/                # 展开与启动
  # 不再默认放 artifact/（迁移期内可并存）
```

---

## 3. `study.yaml` 模版

```yaml
study: mnist_train_size
description: MNIST train_size sweep → test accuracy

experiment:
  name: mnist_linear
  path: examples/experiments/mnist_linear   # 相对仓库根

# 固定条件（并入每次 Run；不进变量轴）
fixed:
  seed: 0
  data:
    name: MNIST
    source: torch                 # stub | torch；unit 测用 stub/Toy
    config:
      batch_size: 64
  model:
    name: linear
  algorithm:
    mode: train
    num_epochs: 2
    lr: 0.1
  system:
    device: cpu

# 变量轴：笛卡尔积展开为多次 Run
axes:
  data.config.train_size: [500, 2000, 8000]

# tags 规则（参与 Run id hash）
tags:
  # 当轴取值匹配时打上 tag
  - when:
      data.config.train_size: 500
    tags: [baseline]

# 每个 Run 的 description 模板（不进 id hash）
run_description: "mnist_linear train_size={data.config.train_size}"
```

填写顺序建议：`study` / `description` → `experiment` → `fixed` → `axes` → `tags`。

---

## 4. Config 约束（摘要）

| 字段 | 进 Run id hash？ | 说明 |
|------|------------------|------|
| 四层 + seed + **tags** | 是 | 改 tag 换目录 |
| `description` | 否 | 给人看 |
| `id` | 否（结果字段） | 由内容算出 |

`data.source`：

| 值 | 行为 |
|----|------|
| `stub` / 缺省且 name 为 Toy | 不下载、假指标（测 Flow） |
| `torch` | 真数据 / 真训（如 MNIST） |

---

## 5. Result 与共享

- Result **只**含可 JSON 化快照（见 [structure.md](code_structure/structure.md) §9.1）  
- 数据集进 `artifact/shared/data/`，各 Run 复用  
- 浏览本 Study 的 Run：**只信 `index.json` + `artifact/runs/`**，不要扫 Experiment 下历史目录  

---

## 6. Flow 阶段名

| 阶段 | 含义 |
|------|------|
| persist | 原 `index`：定稿并写入 `result.json` |
| process | persist 之后：Δ baseline、回填摘要等 |

Study 的 **`index.json`** 仍是编排清单，与 Flow `persist` 无关。

---

## 7. 最小检查清单

- [ ] 已写 `study.yaml`  
- [ ] 已生成 `index.json` 再 launch  
- [ ] 真数据 Run 的 `source: torch` 已显式写出  
- [ ] unit 测未误用真 MNIST  
- [ ] Result 中无 Loader/Module  
