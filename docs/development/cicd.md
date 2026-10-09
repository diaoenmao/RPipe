# CI/CD 接入

本文件是 RPipe 的仓库接入说明，依据 DreamSoul《CI CD 执行规范》和《多人协作代码开发规范》编写。测试怎么写、怎么跑见 [testing.md](testing.md)。当前只有一名维护者，评审人数按这个事实收窄，不另设批准人。

## 一、分支

长期分支只有两个：

| **分支** | **用途** |
| --- | --- |
| `dev` | 日常集成。工作分支只合到这里 |
| `main` | 可发布状态。只接收来自 `dev` 的发布 PR |

临时分支用 `feature/<scope>-<name>`、`fix/<scope>-<name>`、`refactor/<scope>-<name>`。合并后删除。

```text
feature/* / fix/* / refactor/* → PR → dev → 发布 PR → main
```

`dev` 和 `main` 都不能直接 push，不能 force push，也不能删除。合并由 GitHub 在必需检查通过后完成。冲突在工作分支上解决。

当前维护者就是仓库所有者。合入 `dev` 和 `dev → main` 都不要求另一人 Approve。GitHub 不允许作者批准自己的 PR，仓库里又没有第二个人，所以发布 PR 的建议批准人数不启用。必需检查通过后自行合并，也可以打开 Auto-merge，等检查完成后由 GitHub 合并。增加维护者之后，再按改动责任范围打开评审。

工作分支合并后由作者删除。不开启仓库级的自动删除头分支，因为 `dev → main` 的发布 PR 会把长期分支 `dev` 一起删掉。一人维护，暂不使用 CODEOWNERS。

## 二、触发

两条验证工作流监听 `dev` 和 `main` 的 Pull Request，以及合入这两条分支之后的 push。分支方向检查只在 Pull Request 上运行。

| **时机** | **工作流** | **目的** |
| --- | --- | --- |
| 打开或更新 PR | Unit Tests、Package Check、Branch flow | 判断这次改动能不能合进目标分支 |
| 合入 `dev` 或 `main` | Unit Tests、Package Check | 验证实际集成后的提交 |
| 手动 | 本地 `tests/run.py` | 合入前的快速反馈，不代替远端必需检查 |

工作分支直接向 `main` 开 PR 时，Branch flow 失败，合并被阻断。`main` 只接受 head 为 `dev` 的 PR。

## 三、必需检查

`dev` 与 `main` 使用同一组必需检查。分支必须包含目标分支的最新提交。检查名称与工作流里的聚合任务名一致。

| **检查名** | **完成条件** |
| --- | --- |
| `Unit tests` | Ubuntu / Windows × Python 3.10 / 3.13 的矩阵全部成功 |
| `Build package` | Ubuntu / Windows 的 wheel 与 sdist 构建、干净环境安装和 Toy/Stub 流程全部成功 |
| `Branch flow` | PR 的目标是 `dev`，或是从 `dev` 合向 `main` |

必需检查被跳过、取消或没有上报时，不能合并。流程跑完不等于质量通过；失败、超时和未收集到预期用例都阻断。重跑保留原来的失败记录。

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
| 远端检查 | [GitHub Actions](https://github.com/diaoenmao/RPipe/actions)，绑定触发该次运行的提交 |
| 本地测试报告 | `.tmp/test-results/<run_id>/` 的 `manifest.json`、`events.jsonl`、`report.md`。不随 Git clone 提供 |
| 安装包 | 只在 Package Check 的 runner 上构建并当场安装验收，不上传，也不作为 Release 分发 |

对外发布、版本标签和安装包归档还没有接入。把 `dev` 合进 `main` 只表示这条提交通过了上面的必需检查，不等于已经向用户分发。

## 五、当前状态

`dev` 的 Ruleset 从 2026-09-22 起生效。[main Ruleset](https://github.com/diaoenmao/RPipe/rules/24809015) 于 2026-10-10 建立，并成为 `main` 上唯一的分支规则。旧版 branch protection 已删除。两边都要求 PR，禁止 force push 和删除，没有人可以绕过，必需检查是 `Unit tests`、`Build package` 和 `Branch flow`，并且分支要包含目标分支的最新提交。仓库已打开 Auto-merge。

受控失败见 [PR #20](https://github.com/diaoenmao/RPipe/pull/20)：`tmp/branch-flow-probe` 直接合向 `main` 时，[Branch flow](https://github.com/diaoenmao/RPipe/actions/runs/37976079219) 失败，合并状态为 BLOCKED。该 PR 已关闭，分支已删除。同一次打开的 Unit Tests 和 Package Check 在确认阻断后取消，不作为通过或失败依据。
