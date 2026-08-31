# Tests

测试目录与标签遵循 [TESTING.md](../docs/TESTING.md)，并与 [LAYOUT.md](../docs/LAYOUT.md) §6、[CODE_STRUCTURE.md](../docs/CODE_STRUCTURE.md) §4 一致。

## 原则

- **镜像源码**：`tests/rpipe/` ↔ `src/rpipe/`
- **层级是标签**：`unit` / `integration` / `e2e`，不设同名一级目录
- **强制三维标签**：Level × Type（`location`|`content`|`physical`）× Priority（`p1`|`p2`|`p3`）
- **辅助目录**：`_fixtures/`、`_helpers/`（辅助函数不得 `test_` 前缀）
- **包外 Study**：e2e 指向仓库根 `studies/`（不再有 `tests/examples/`）

## 排除不镜像项

`__pycache__/`、`.pytest_cache/`、`.egg-info/`、`output/`、`.test-results/`、虚拟环境、大数据 asset。

## Markers（须在 pyproject 注册）

`unit`、`integration`、`e2e`、`location`、`content`、`physical`、`p1`、`p2`、`p3`，以及 `structure_layer`、`flow_layer`、`artifact_layer`、`application_layer`、`runtime`、`memory`、`slow`、`external` 等。

## 推荐筛选

```text
pytest -m "unit and p1 and not slow and not external"
pytest -m "integration and p1"
pytest -m "e2e"
```

结果目录约定：`.test-results/<run_id>/`（`manifest.json` + `events.jsonl`）。
