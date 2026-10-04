# 仓库整理与 main 迁移验收

> 本文与对应 JSON 是2026-10-04发布前整理验收的固定快照，文件哈希与“未提交/合并”指向当时输入。后续 README/新图与阶段发布见 [v0.3.0](releases/v0.3.0.md)，不要将旧快照当作当前分支状态。

## 一、目标与范围（2026-10-04）

用户要求更新、清理和整理整个仓库，并判断下一步能否合并 dev、替换现有 main。本轮整理公开入口、文档导航、研究成果、测试骨架和 CI / 打包验证；以最新远端分支及实际检查结果判断迁移条件。

执行前 `git fetch origin` 成功。HEAD / dev / origin/dev 为 `8bccbac321d4c3ac1ea9892a5e774c114e0298c6`，main / origin/main 为 `98648f3a5c7db7dccf3ca806410d5b6fdee9484c`；main 是 dev 的祖先，左右计数为 0 / 61。现有源码与新增研究成果的来源保持分开。

## 二、整理约定

1. 更新 README、SUMMARY、BRAINSTORM 和 Study 导航，加入本机完整历史曲线和受控现代 main 对照。过去的失败、限制及日期快照保留，不能倒写为后来的结果。
2. 给历史 Study 写明专用 Registry 入口、准备数据与依赖的方法；通用 `python -m rpipe` 不自动注册这个归档配方。
3. 正式配置、重跑与审计程序、JSON / Markdown 报告和图保留。训练数据、checkpoint、日志、临时运行环境及科学失败证据不删除；不会把运行中的临时写入标记加入 Git。
4. 只移除已有真实测试文件或子目录支撑的冗余 `.gitkeep`；空镜像目录的占位文件保留，源码 / 测试目录合同继续成立。
5. 补验 native CPU 的 unit / integration / CLI 流程与 wheel / sdist 交付，区分实际本机通过和远端 CI 待执行。必要的 CI 调整遵循现有测试标签和成本边界，不重新运行长训练。
6. 写清旧 main 命令、配置、模型 API 与 checkpoint 的迁移方法；不把目录格式部分兼容说成原 main checkpoint 可直接恢复。

## 三、完成条件

- 根目录、docs、studies 的公开入口与实际树一致；最新结果可从首页到达。
- 冗余资源处理有明确清单，科学数值源、配置与绑定证据哈希未变化。
- 本机所需测试、安装/CLI及产物检查完成，有可复核结果；未执行的远端或平台门明确记录。
- dev → main 的真实分支关系、行为变化、已满足条件和剩余发布步骤有具体结论。

### 2026-10-04 测试入口与安装诊断

原统一测试入口在当前沙箱中实际为 283 passed / 3 failed，三项缺少注入的 Kornia 依赖，原结果保留。正常本机进程执行同一入口为 286 passed / 3 deselected、退出0。最小对照进一步确认当前沙箱嵌套子进程的隐式环境/标准输出传递丢失；显式传递后原失败用例通过。本轮给测试入口显式传递环境和 stdout/stderr，并补真实子进程的诊断/参数/退出码回归，不把现象扩称一般 Windows/Conda 缺陷。

初次安装后 wheel 模块入口通过，Windows console script 启动遇到 WinError32 共享冲突；原失败工作区保留。包验收只对 CreateProcess 尚未成功的共享冲突有限重试，不重跑已经启动后失败的命令。

实际 main 合并或强制覆盖不属于本轮已执行动作；本轮完成可审查的迁移准备并回答合并条件。

## 四、实际结果

整理已完成；详细机器可读记录见 [REPOSITORY_CLEANUP_RESULT.json](REPOSITORY_CLEANUP_RESULT.json)。

| **范围** | **完成内容** |
| --- | --- |
| 首页与开发文档 | README、SUMMARY、BRAINSTORM、LAYOUT 和 Study 指南更新为当前状态；历史日期快照、失败与限制保留 |
| Study 导航 | 新增 [14项 Study 目录](../studies/README.md)、[历史复现入口](../studies/main_historical/README.md)，明确自定义 Registry、准备顺序和本地产物不会随 clone 提供 |
| 测试骨架 | 删除13个已有文件/子目录支撑的冗余 `.gitkeep`，保留7个空镜像目录占位；没有删除测试或生产代码 |
| 本地事务文件 | `studies/**/docs/*.tmp` 和 `*.claim` 加入忽略；原文件与 SHA 保留，不忽略正式 JSON / Python 报告 |
| 构建副产物 | 移除本轮生成的根 `build/`，其91个文件删除前已逐个确认为生产源码副本；发行包及验证记录保存在 `.tmp/repo-cleanup/` |
| 统一测试入口 | 修复当前沙箱嵌套进程的环境和输出传递，新增两个真实子进程回归；生产数值源未修改 |
| 包与 CI | 新增安装后公开 CLI 验收程序；CI 增加 Windows、Python3.13 CPU门，wheel/sdist 各自隔离安装并执行 CLI / Toy 合同 |
| 报告模板 | 未生成曲线的链接改为代码示例，避免模板默认存在无效图片链接 |
| main 迁移 | 新增 [MAIN_MIGRATION.md](MAIN_MIGRATION.md)，列明旧命令、配置、模型 API、Stats精度、std及 checkpoint/resume 的变化 |

| **验证** | **结果** |
| --- | --- |
| 统一 CPU 执行门 | 288 passed / 3 deselected、退出0，20.14秒；c1/c2、非 external/gpu/slow 的 unit / integration 与两个新入口回归 |
| wheel / sdist | 两种发行包构建、安装后各9命令通过；各4 Run / 2 Experiment完整，配置不变、重复 launch 跳过成功任务 |
| 包内容 | wheel包含全部91个生产 Python 文件，内容一致；没有打包本地研究 Run、数据与 `.tmp` |
| 科学依据 | 772个唯一证据文件重新核验、错误0；历史99源+65计划+16raw及17/465项最终绑定、现代来源和56项CPU存档绑定一致；没有改门限或重做长训练 |
| 仓库静态检查 | 79份Markdown、977个有效相对链接，正式路径缺失0；281个缺失本地忽略产物及1个模板占位单独标记；Python语法、两份CI YAML和 `git diff --check` 通过 |
| 远端 CI | 已推送 `8bccbac` 的两项旧工作流通过；本轮改进工作流尚未提交运行，不能记为新矩阵通过 |

本机打包验收复用了现有科学依赖，未执行全新网络依赖解析。完整原始命令、退出码、首轮失败及安装记录保留在 `.tmp/repo-cleanup/`；这些临时文件不随 clone 提供。正式 JSON 保存摘要与哈希，科学报告本身保留完整边界。

## 五、下一步可以开始 dev → main 迁移 PR

当前 main 为 dev 祖先，没有 main 独有提交，具备开始迁移 PR 的分支条件。研究复现与本轮 CPU / 包功能门已经完成；这是新接口与产物格式的迁移发布。

发布前还需把当前本地新增复跑入口、正式报告、图、清单及整理改动纳入提交，推送后等待新 CI 的 Ubuntu / Windows、Python3.10 / 3.13 与隔离安装门通过，实际合并时再次核对远端 refs。只合并现在已推送的 dev，会遗漏本机尚未提交的新成果。

旧 main checkpoint 不能直接作为新 Run resume，旧脚本调用也需要迁移；旧提交和原产物应保留为读取与回退依据。具体字段和操作顺序见 [迁移说明](MAIN_MIGRATION.md)。本轮未实际提交、推送或合并 main。
