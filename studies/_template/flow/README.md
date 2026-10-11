# flow/

这些目录是可选的 Study 扩展位置，不是库默认阶段的副本。没有 hook 时仍执行 `src/rpipe/flow/` 的默认链，不必填写空函数。

有研究特有的准备或检查时，在对应阶段的 `run.py` 定义 `run(ctx)`；`prepare/run.py` 还可定义 `before(ctx)`。同目录 `__init__.py` 只导出这些函数，不写实现，并在 `study.yaml` 启用 `flow.study_phases: true`。

helper 放独立模块，手动工具不自动执行。完整目录和示例见 [Flow 规范](../../../docs/code/flow.md#140-study-阶段目录)。
