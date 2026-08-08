# RPipe 测试报告

- 生成时间：2026-08-08 23:32:52
- 规范依据：研讨纪要 测试规范（unit / integration / e2e × location / content / physical × p1–p3）
- CUDA：False（cpu）
- torch：2.13.0+cpu

## 1. 总览

| 指标 | 数量 |
|------|------|
| Total | 28 |
| Pass | 28 (100.0%) |
| Fail | 0 (0.0%) |
| Skip | 0 |

## 2. 按优先级（Priority）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `p1` | 21 | 0 | 0 | 21 | 100.0% |
| `p2` | 7 | 0 | 0 | 7 | 100.0% |

## 3. 按测试层级（Level）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `e2e` | 2 | 0 | 0 | 2 | 100.0% |
| `integration` | 2 | 0 | 0 | 2 | 100.0% |
| `unit` | 24 | 0 | 0 | 24 | 100.0% |

## 4. 按测试类型（Type）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `content` | 21 | 0 | 0 | 21 | 100.0% |
| `location` | 5 | 0 | 0 | 5 | 100.0% |
| `physical` | 2 | 0 | 0 | 2 | 100.0% |

## 5. 按架构层（Architecture Layer）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `algorithm_layer` | 4 | 0 | 0 | 4 | 100.0% |
| `application_layer` | 2 | 0 | 0 | 2 | 100.0% |
| `config_layer` | 6 | 0 | 0 | 6 | 100.0% |
| `data_layer` | 4 | 0 | 0 | 4 | 100.0% |
| `model_layer` | 4 | 0 | 0 | 4 | 100.0% |
| `plugins_layer` | 7 | 0 | 0 | 7 | 100.0% |
| `schema_layer` | 4 | 0 | 0 | 4 | 100.0% |
| `system_layer` | 4 | 0 | 0 | 4 | 100.0% |

## 6. 按模块 Tag（module_*）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `module_provider` | 4 | 0 | 0 | 4 | 100.0% |
| `module_registry` | 2 | 0 | 0 | 2 | 100.0% |
| `module_trainer` | 2 | 0 | 0 | 2 | 100.0% |
| `untagged` | 20 | 0 | 0 | 20 | 100.0% |

## 7. 按功能 Tag（feature_*）

| 维度 | Pass | Fail | Skip | Total | Pass率 |
|------|------|------|------|-------|--------|
| `feature_providers` | 2 | 0 | 0 | 2 | 100.0% |
| `feature_smoke` | 1 | 0 | 0 | 1 | 100.0% |
| `untagged` | 25 | 0 | 0 | 25 | 100.0% |

## 8. 集成 / 端到端

| Level | 测试 | 结果 | 时长(s) | markers |
|-------|------|------|---------|---------|
| `e2e` | `test_e2e_smoke_pipeline_prepare_and_train` | **passed** | 7.118093400000362 | algorithm_layer, application_layer, content, data_layer, e2e, feature_smoke, gpu, model_layer, p1, slow, system_layer |
| `e2e` | `test_e2e_device_selection_reported` | **passed** | 0.00012380000043776818 | application_layer, e2e, gpu, p2, physical, runtime |
| `integration` | `test_runtime_providers_wire_into_native_trainer` | **passed** | 0.0007322000019485131 | config_layer, content, feature_providers, integration, module_trainer, p1, plugins_layer, system_layer |
| `integration` | `test_native_data_and_model_provider_build_chain` | **passed** | 0.0003947000004700385 | content, data_layer, integration, model_layer, p1 |

## 9. 失败用例与定位

无失败。

## 10. 后续建议

- 本轮全部通过；可将 `external`/`slow` 范围扩大到可选第三方冒烟。
- 当前环境 **无 CUDA**；GPU physical / e2e 已按 CPU 回退或 skip。有 GPU 时重跑 `pytest -m "gpu or e2e"`。
- 结果落盘：`output/test_results/results.jsonl`；可重复解析生成本报告。

## 附录：全部用例

| Outcome | Priority | Level | Type | Name |
|---------|----------|-------|------|------|
| passed | p1 | e2e | content | `test_e2e_smoke_pipeline_prepare_and_train` |
| passed | p2 | e2e | physical | `test_e2e_device_selection_reported` |
| passed | p1 | integration | content | `test_runtime_providers_wire_into_native_trainer` |
| passed | p1 | integration | content | `test_native_data_and_model_provider_build_chain` |
| passed | p1 | unit | content | `test_native_metric_provider_make_metric` |
| passed | p2 | unit | content | `test_lm_eval_lists_gsm8k_without_requiring_install` |
| passed | p1 | unit | content | `test_build_runtime_cfg_returns_runtime_config` |
| passed | p1 | unit | content | `test_apply_control_and_hyper_overrides` |
| passed | p1 | unit | content | `test_registry_register_build_and_keys` |
| passed | p2 | unit | content | `test_global_model_dataset_registries_nonempty_after_imports` |
| passed | p1 | unit | content | `test_native_data_provider_lists_builtin_sets` |
| passed | p1 | unit | content | `test_datasets_provider_package_and_list` |
| passed | p1 | unit | content | `test_make_linear_model_forward_shape` |
| passed | p2 | unit | physical | `test_linear_forward_runtime_budget` |
| passed | p1 | unit | content | `test_builtin_providers_registered_by_package_name` |
| passed | p1 | unit | content | `test_provider_package_attribute_aligned` |
| passed | p1 | unit | content | `test_metric_modes_online_vs_benchmark` |
| passed | p1 | unit | content | `test_native_providers_available` |
| passed | p2 | unit | content | `test_optional_providers_report_availability_without_raising_on_list` |
| passed | p1 | unit | content | `test_result_blob_schema_accepts_valid` |
| passed | p1 | unit | content | `test_result_blob_schema_rejects_wrong_schema_id` |
| passed | p1 | unit | content | `test_manifest_schema_accepts_and_rejects` |
| passed | p2 | unit | location | `test_schema_ids_stable` |
| passed | p1 | unit | content | `test_trainer_proxy_dispatches_native` |
| passed | p1 | unit | location | `test_src_rpipe_package_layout_exists` |
| passed | p1 | unit | location | `test_plugin_layer_modules_exist` |
| passed | p1 | unit | location | `test_tests_mirror_unit_dirs_exist` |
| passed | p2 | unit | location | `test_system_backend_files_exist` |
