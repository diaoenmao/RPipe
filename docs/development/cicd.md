# CI/CD 接入

本文件是 RPipe 的仓库接入说明，依据 DreamSoul《CI CD 执行规范》和《多人协作代码开发规范》编写。测试怎么写、怎么跑见 [testing.md](testing.md)。当前只有一名维护者，评审人数按这个事实收窄，不另设批准人。

## 一、分支

长期分支只有两个：

| **分支** | **用途** |
| --- | --- |
| `dev` | 日常集成。工作分支只合到这里 |
| `main` | 可发布状态。只接收来自 `dev` 的发布 PR |

临时分支用 `feature/<scope>-<name>`、`fix/<scope>-<name>`、`refactor/<scope>-<name>`，Codex工作分支默认 `codex/<name>`。合并后删除。

```text
feature/* / fix/* / refactor/* → PR → dev → 发布 PR → main
```

`dev` 和 `main` 都不能直接 push，不能 force push，也不能删除。合并由 GitHub 在必需检查通过后完成。冲突在工作分支上解决。

当前维护者就是仓库所有者。合入 `dev` 和 `dev → main` 都不要求另一人 Approve。GitHub 不允许作者批准自己的 PR，仓库里又没有第二个人，所以发布 PR 的建议批准人数不启用。必需检查通过后自行合并，也可以打开 Auto-merge，等检查完成后由 GitHub 合并。增加维护者之后，再按改动责任范围打开评审。

### 分支管理（2026-10-15）

1. **同一工作流同一时间只保留一条工作分支。** 同一主题的后续工作在原分支续做，不另开同质分支；需要从最新 `dev` 重新开始时，新开分支前核对旧分支提交与本地改动已保全，并删除旧工作分支，不并存同主题分支。不得覆盖用户改动或删除其他工作流的活动分支。
2. **合入后立即清理。** PR 合进 `dev`，且最终工作提交与实际 `dev` 的必需 CI 全部通过后，立即删除对应远程和本地工作分支。先核对提交已纳入 `dev`、没有未保全本地改动，再执行普通删除；不以强制删除代替核验，不删除 Run、数据或 provenance。
3. **长期分支只有 `main` 和 `dev`。** 两者持续保护，禁止直接 push、force push 和删除；工作分支不能变相长期积累。
4. **建议开启合并后自动删除工作分支。** 同时核对 `main` / `dev` 的删除保护，以及 `dev → main` 发布 PR 的头分支保全，不能把长期分支当作临时头分支清理。自动删除远程分支不代替本地分支清理、未完成 CI 核对或提交保全核验。

5. **每约 5 个 PR 做一次全面清理。** 从最近一次清理基线起，按 GitHub 实际合入 dev 的 PR 数累计；达到约 5 个时执行本地/远程分支、已登记 worktree、已处理 PR/Issue 的盘点。已合但不用的分支与废弃 worktree 先核对提交/未提交内容和 GPU 使用情况，再按权限清理；未合分支即使过时也仅在维护 PR 说明列清单，不删除。已处理且验收完成的 PR/Issue 可按本轮授权关闭，不等待发版；未完成事项不因清理关闭。清单、保留理由与基线写入 record，临时核验保留 .tmp；后续每次合入检查累计数，不为此增加定时弹窗或改动运行架构。

2026-10-16 最新维护授权（13:52 指令优先）已撤销此前“保留已合分支”的阶段限制：已合 dev 的本地/远程分支，在提交保全与实际 CI 验收后清理；未合分支保留。GPU 任务使用中的分支/worktree 不动，不删除实验数据、结果或 provenance，不 force push、不动 main。GitHub 自动删除设置维持现状；不用强制删除代替祖先与工作区检查。一人维护，暂不使用 CODEOWNERS。


### 下游项目的分支与可选 CI

使用源码模板的研究项目沿用 `main` / `dev` 的职责：工作分支从 dev 建立，通过必要验证和 PR 合回 dev；稳定后再由 dev 通过发布 PR 进入 main。新仓库没有 dev 时，从新仓库的 main 建立，使两者拥有共同历史，不从无关模板分支拼接历史。

GitHub Actions 可按项目需求启用，也可保持关闭。启用时配置真实有效的检查和资源边界；关闭时保留本地工程及研究验证、记录结果与失败，不自动恢复 workflow，也不把未执行说成通过。分支规则若仍要求不会上报的检查，会使 PR 等待；调整这些规则需要对应仓库的明确授权。RPipe 本仓库继续使用下文的已启用 Actions 和必需检查，不因下游选择而放宽。

## 二、触发

两条验证工作流监听 `dev` 和 `main` 的 Pull Request，以及合入这两条分支之后的 push。分支方向检查只在 Pull Request 上运行。

| **时机** | **工作流** | **目的** |
| --- | --- | --- |
| 打开或更新 PR | Unit Tests、Package Check、Branch flow | 判断这次改动能不能合进目标分支 |
| 合入 `dev` 或 `main` | Unit Tests、Package Check | 验证实际集成后的提交 |
| 手动 | 本地 `tests/run.py` | 合入前的快速反馈，不代替远端必需检查 |

工作分支直接向 `main` 开 PR 时，Branch flow 失败，合并被阻断。`main` 只接受 head 为 `dev` 的 PR。

普通工作分支的 push 不触发这些工作流，不代表 Actions 已关闭。当前没有 `workflow_dispatch` 手动触发入口；表中的“手动”指本地测试。触发条件在 `.github/workflows/*.yml`，仓库总开关在 `Settings → Actions → General → Actions permissions`，单个工作流可在 Actions 页面停用。公开 / 私有不是启用开关，组织策略还可能限制仓库设置，见 [GitHub Actions 设置](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)。

## 三、必需检查

`dev` 与 `main` 使用同一组必需检查。分支必须包含目标分支的最新提交。检查名称与工作流里的聚合任务名一致。

| **检查名** | **完成条件** |
| --- | --- |
| `Unit tests` | Ubuntu / Windows × Python 3.10 / 3.13 的矩阵全部成功 |
| `Build package` | Ubuntu / Windows 的 wheel 与 sdist 构建、干净环境安装和 Toy/Stub 流程全部成功 |
| `Branch flow` | PR 的目标是 `dev`，或是从 `dev` 合向 `main` |

本项目的 `Unit tests` / `Build package` 聚合任务使用 `if: always()`，只接受依赖矩阵的结果为 `success`；矩阵失败、取消或被跳过时，聚合门失败。GitHub 本身可能接受检查的 `skipped` / `neutral` 状态，不能把“配置了必需检查”等同于“一定执行了全部用例”，见[必需检查说明](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)。未触发、未上报或未完成的必需检查不能证明通过；本项目不以跳过聚合门绕过验证。流程跑完不等于质量通过，重跑保留原来的失败记录。

本地对应命令：

```bash
python tests/run.py --core
python tests/run.py --all --cost-class c1 --cost-class c2 -- -m "(integration or e2e) and not external and not gpu and not slow"
```

Package Check 在 Actions 里执行 `python -m build`，再在独立虚拟环境安装 wheel 和 sdist，并运行 `tests/package_smoke.py`。

支持平台是 GitHub 托管的 Ubuntu 与 Windows。不覆盖 GPU 和可选的 HF 依赖。

## 四、结果与产物

| **结果** | **位置** |
| --- | --- |
| 远端检查 | [GitHub Actions](https://github.com/DreamSoul-AI/RPipe/actions)，绑定触发该次运行的提交 |
| 本地测试报告 | `.tmp/test-results/<run_id>/` 的 `manifest.json`、`events.jsonl`、`report.md`。不随 Git clone 提供 |
| 安装包 | 只在 Package Check 的 runner 上构建并当场安装验收，不上传，也不作为 Release 分发 |

没有自动发布标签 / Release、PyPI 或安装包归档的工作流。把 `dev` 合进 `main` 不自动发布版本；维护者明确授权后可按第六节手工创建标签与 GitHub Release。

## 五、当前状态

2026-10-13 复核：RPipe 是公开模板仓库，仓库级 Actions 已开启，`Unit Tests`、`Package Check` 和 `Branch flow` 均为 active。v0.3.0 的实际 main 提交 `804b3823` 的 [Unit Tests](https://github.com/DreamSoul-AI/RPipe/actions/runs/38293625374) 和 [Package Check](https://github.com/DreamSoul-AI/RPipe/actions/runs/38293625406) 均通过；这是该提交的 CPU / 安装包验证，不替代下游项目或异机实验验收。

`dev` 的 Ruleset 从 2026-09-22 起生效。[main Ruleset](https://github.com/DreamSoul-AI/RPipe/rules/24809015) 于 2026-10-10 建立，并成为 `main` 上唯一的分支规则。旧版 branch protection 已删除。两边都要求 PR，禁止 force push 和删除，没有人可以绕过，必需检查是 `Unit tests`、`Build package` 和 `Branch flow`，并且分支要包含目标分支的最新提交。仓库已打开 Auto-merge。

受控失败见 [PR #20](https://github.com/DreamSoul-AI/RPipe/pull/20)：`tmp/branch-flow-probe` 直接合向 `main` 时，[Branch flow](https://github.com/DreamSoul-AI/RPipe/actions/runs/37976079219) 失败，合并状态为 BLOCKED。该 PR 已关闭，分支已删除。同一次打开的 Unit Tests 和 Package Check 在确认阻断后取消，不作为通过或失败依据。

### 模板派生仓库

本文件描述 RPipe 的分支和 CI，不自动约束下游项目。从模板创建仓库会复制目录和文件，包括 `.github/workflows/` 的触发条件；仓库级 Actions 开关、单个工作流停用状态、Ruleset、Secrets 和 Auto-merge 不是这些文件的一部分，需要在新仓库单独核对。新仓库还受所属组织策略约束，原仓库关闭 Actions 不等于模板派生仓库也关闭。

1. 新项目按自己的分支方案选择保留、调整或移除复制的 workflow。若沿用 dev → main，从新仓库的 main 建立 dev，使两者拥有共同历史；不要依靠模板的“Include all branches”，其分支历史互不相关，不能正常合 PR，见[模板说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)。
2. 分别配置 Actions 和分支规则。若明确关闭自动 CI，保留本地验证入口，并检查是否仍要求不会运行的必需检查，避免 PR 长期等待；规则调整需要该仓库的明确授权。
3. 下游项目复制固定版本的 `src/rpipe/` 并沿用 structure/flow 架构，不需要复制整个仓库或 CI；接入步骤见 [README](../../README.md#在其他项目中使用-rpipe)。继续 RPipe 自身的历史对照则 clone 原仓库，保留 Study 需要的历史 Git 对象，不使用模板新建仓库替代。

## 六、维护者授权后的版本发布

1. 以最近正式标签为基线核对完整差异，选择未使用的新版本号，在 `docs/releases/<version>.md` 编写完整报告，区分新增工作、迁移、验证与未覆盖范围。
2. 同步 pyproject.toml、rpipe.__version__ 与 README；通过工作 PR 进入 dev，重新执行当前提交的必需检查。
3. 更新或创建 dev → main 发布 PR，等待 Unit tests / Build package / Branch flow 全部成功再合并；不删除 dev，不直接 push main。
4. 等待实际 main 集成提交的 CPU / 包 CI 通过，再在该确切提交上建立新 annotated tag，推送标签并创建 GitHub Release；不能移动或覆盖旧标签 / Release。
5. Release 正文使用入库报告，仓库相对链接转换为指向发布提交的 GitHub 链接，并注明确切提交 / PR / CI。默认仅 GitHub 源码归档，不上传本机数据、Run、缓存或安装包；PyPI / 二进制分发需另行授权。

当前版本报告入口：[v0.4.1](../releases/v0.4.1.md)，前一正式报告为 [v0.4.0](../releases/v0.4.0.md)。手工发布不引入自动发版工作流，历史实测源码与哈希不因包版本号改变而改写；正式标签与最终状态以 GitHub Release 为准。
