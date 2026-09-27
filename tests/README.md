# Tests

测试目录与标签遵循 [TESTING.md](../docs/TESTING.md)（2026-09-28），并与 [LAYOUT.md](../docs/LAYOUT.md) 一致。

## 原则

- **镜像源码**：`tests/rpipe/` ↔ `src/rpipe/`。不设 `tests/unit/`、`tests/e2e/` 一级目录。
- **层级是标签**：每项测试恰好一个 `unit` / `integration` / `e2e`。
- **强制标签**：Level × Objective（`location`|`content`|`physical`）× Priority（`p1`|`p2`|`p3`）。
- **成本与结果**：恰好一个 `cost(cost_class=...)`，恰好一个 `result_type(..., detail=...)`。缺则收集失败。
- **e2e**：系统入口是 `flow/cli.py`，用例放在 `tests/rpipe/flow/`，标签为 `e2e`。集成和端到端用例的文档字符串写明起点、终点和预期。

## 排除不镜像项

`__pycache__/`、`.pytest_cache/`、`.egg-info/`、`.tmp/`、虚拟环境、Study 生成的 `runs/` / `shared/` / `scripts/`。

## 已登记 markers

强制：`unit`、`integration`、`e2e`、`location`、`content`、`physical`、`p1`、`p2`、`p3`。

元数据：`cost`（`c1`–`c4`）、`result_type`（`categorical`|`numeric`，`detail` 为 `summary`|`metrics`|`samples`）。

架构层：`structure_layer`、`flow_layer`。

模块：`module_api`、`module_control`、`module_data`、`module_model`、`module_algorithm`、`module_system`、`module_artifact`、`module_make`、`module_cli`、`module_runner`、`module_process`。

资源与执行特征：`runtime`、`memory`、`gpu`、`slow`、`external`、`flaky`、`quarantined`。

## 预算起点

C1：≤ 1 s 且 ≤ 256 MiB，无真实 GPU、无外网。C2：超出 C1，≤ 60 s 且 ≤ 2 GiB。C3：真实 GPU 或超出 C2。C4：批量基准。真实 GPU 同时标 `gpu`。

## 执行入口

```text
python tests/run.py --fast
python tests/run.py --core
python tests/run.py --level unit --priority p1 --cost-class c1
python tests/run.py --cost-class c3
python tests/run.py --all
```

每次运行写入 `.tmp/test-results/<run_id>/manifest.json`、`events.jsonl` 和 `report.md`。
