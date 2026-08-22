# Results: mnist_train_size

> 对应计划：[PLAN.md](PLAN.md)  
> 跑通日期：2026-08-22  
> Experiment：`mnist_linear`（真实 MNIST + linear）

## 1. 结论（一句话）

在固定 2 epoch / SGD lr=0.1 / seed=0 下，**增大 `train_size` 明显提高测试准确率**：500 → 2000 → 8000 对应 accuracy **0.764 → 0.850 → 0.893**。

## 2. Runs

| train_size | tags | Run id | status | test accuracy | last-batch loss |
|------------|------|--------|--------|---------------|-----------------|
| 500 | `baseline` | `03d885fff93ebdb8` | succeeded | 0.7642 | 0.415 |
| 2000 | — | `05d8fd2bd9e28389` | succeeded | 0.8495 | 0.384 |
| 8000 | — | `219dd92975be42ed` | succeeded | 0.8928 | 0.275 |

相对 baseline（500）：

| train_size | Δ accuracy |
|------------|------------|
| 2000 | +0.0853 |
| 8000 | +0.1286 |

产物：

- Study index：`examples/studies/mnist_train_size/index.json`
- Results：`examples/experiments/mnist_linear/artifact/<id>/result.json`

## 3. 走完流程时踩到的卡点（给壳子优化用）

这些是这次「真实验」暴露的问题，比 stub smoke 更有信息量：

### 3.1 Result 序列化炸了（已修）

- **现象**：`summarize` 把 `state['data']` / `model` 原样塞进 Result → `DataLoader` / `nn.Module` 无法 `json.dump`
- **位置**：`flow/summarize` → `artifact.result.write_result`
- **含义**：Runtime handle 与 Result 契约没有边界；真 Structure 一落地就爆
- **已做**：summarize 只快照可 JSON 字段，丢掉 `train_loader` / `test_loader` / `module`
- **壳子方向**：约定「state 可持有 runtime；Result 只收可序列化投影」；最好有统一 `to_result_snapshot()`

### 3.2 数据缓存在 Run Artifact 里，重复下载

- **现象**：每个 `artifact/<id>/assets/mnist/` 各自 `download=True`，三次 Run 重复拉 MNIST
- **含义**：Asset 按 Run 隔离合理，但**共享只读数据**没有 Experiment / 全局 cache 约定
- **壳子方向**：`data.path` 或 Experiment 级 cache 根；prepare 优先复用

### 3.3 stub → 真训练后，旧 unit 测变慢 / 变脆

- **现象**：`tests/unit/flow/test_runner.py` 仍用 `data.name: MNIST`，一跑就下载 + 真训
- **含义**：库内默认语义从 stub 切到真路径后，测试没有「Toy / stub」显式开关
- **已做**：unit 改用 `Toy` 走 stub；e2e / Study 走真 MNIST
- **壳子方向**：`source: stub | torch` 或仅当显式 `train_size` / adapter 时走真路径

### 3.4 Study 与 Experiment 配置职责仍手工

- **现象**：train_size 轴、baseline tag、`num_epochs` 都在 Study `run.py` 里手写 patch；grid 刚支持 `patches=`
- **含义**：编排契约能用，但缺「声明式 Study 配置」（YAML 变量轴）时，复制粘贴成本高
- **壳子方向**：可选 `study.yaml`：axes / tags / baseline 规则 → 生成 patches

### 3.5 同 Experiment 下旧 Artifact 污染浏览

- **现象**：`artifact/` 里仍有历史 stub / 其它 Study 的 Run；扫目录不等于本 Study
- **含义**：统一 `index.json` 很必要；只看 `artifact/*` 会混
- **壳子方向**：消费方强制经 Study index；或 Artifact 元数据带 `study` 字段

### 3.6 比较 baseline 仍靠人读表

- **现象**：有 `tags: [baseline]`，但没有一键 Δ 工具；RESULTS 表是手写脚本扫的
- **壳子方向**：薄函数 `compare_to_tag(index, tag='baseline', metric='accuracy')`

### 3.7 importlib 加载 examples 模块别扭

- **现象**：Study 用 `spec_from_file_location` 加载 grid/launch，路径脆弱
- **壳子方向**：examples 可安装入口，或约定 `python -m` 包布局

## 4. 流程回放（是否按架构走）

1. 写 `PLAN.md`（本 Study 契约）  
2. 扩展 Structure（data/model/train）+ grid `patches`  
3. `run.py`：expand → **先写 `index.json`** → launch  
4. 收 Result → 写本 `RESULTS.md`  

与 CONCEPT「先编排清单、再执行」一致；卡点集中在 **真 runtime ↔ Result/Asset 边界** 与 **数据缓存层级**。

## 5. 建议的下一刀（已吸收进权威文档）

下列方向已写入 CONCEPT / LAYOUT / structure.md §9.1 / flow 分册 / [STUDY_GUIDE.md](../../../docs/STUDY_GUIDE.md)，**代码迁移尚未完成**（当前 examples 仍可能把 Artifact 放在 Experiment 下）：

1. **Result 快照契约** — state 可持有 runtime；Result 只收可 JSON 投影（已在 summarize 止血；待 `to_result_snapshot`）  
2. **Artifact 挂 Study** — `shared/data|model` + `runs/<id>/`；解决重复下载与目录污染（3.2 / 3.5）  
3. **`source: stub|torch`** — 避免 unit 误触真数据（3.3）  
4. **`study.yaml` 模版** — 先填配置再展开（3.4）；见 `examples/studies/_template/study.yaml`  
5. **Flow：`index` → `persist`，其后加 `process`** — persist 负责序列化落盘；process 做 Δ / 回填（3.6）  
