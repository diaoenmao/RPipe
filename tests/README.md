# Tests

测试目录与标签遵循 [TESTING.md](../docs/TESTING.md)，并与 [LAYOUT.md](../docs/LAYOUT.md) §6、[CODE_STRUCTURE.md](../docs/CODE_STRUCTURE.md) §4 一致。

## 原则

- **镜像源码**：`tests/rpipe/` ↔ `src/rpipe/`。不设 `tests/unit/`、`tests/e2e/` 一级目录。
- **层级是标签**：每项测试恰好一个 `unit` / `integration` / `e2e`。
- **强制三维标签**：Level × Type（`location`|`content`|`physical`）× Priority（`p1`|`p2`|`p3`）。缺标则收集失败。
- **辅助目录**：`_fixtures/`、`_helpers/`（辅助函数不得 `test_` 前缀）。
- **e2e**：系统入口是 `flow/cli.py`，用例放在 `tests/rpipe/flow/`，标签为 `e2e`。

## 排除不镜像项

`__pycache__/`、`.pytest_cache/`、`.egg-info/`、`.test-results/`、虚拟环境、Study 生成的 `runs/` / `shared/`。

## 已登记 markers

强制：`unit`、`integration`、`e2e`、`location`、`content`、`physical`、`p1`、`p2`、`p3`。

架构层：`structure_layer`、`flow_layer`。

模块：`module_api`、`module_control`、`module_data`、`module_model`、`module_algorithm`、`module_system`、`module_artifact`、`module_make`、`module_cli`、`module_runner`、`module_process`。

执行特征：`runtime`、`memory`、`slow`、`external`、`flaky`、`quarantined`。

## 执行入口

```text
python tests/run.py --fast
python tests/run.py --core
python tests/run.py --level unit --priority p1
python tests/run.py --level integration
python tests/run.py --level e2e
python tests/run.py --all
python tests/run.py --layer structure_layer --type content
```

等价 pytest：

```text
pytest -m "unit and p1 and not slow and not external"
pytest -m "unit and not slow and not external"
pytest -m "e2e"
```

每次运行写入 `.test-results/<run_id>/manifest.json` 与 `events.jsonl`。
