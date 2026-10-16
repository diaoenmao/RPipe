# 研究计划正本与译本

## 一、文件与权威

按 Enmao 2026-10-16 指示，将 DEM 暂存交付的两份中文正本**按字节复制**到此处，保留中文文件名、编码、换行、日期、原文链接和内容；未把 RPipe 适配规则写回正本。当前规范 184 行、模板 122 行，原文更新日期均为2026-10-16。中文正本优先，英文为完整内容翻译与必要的本仓库导航说明，不作为另一套规范。

| **文件** | **字节数** | **SHA-256** |
| --- | --- | --- |
| [研究计划规范 Research Plan Standard.md](<研究计划规范 Research Plan Standard.md>) | 25618 | `61fc9fbe1130810b41ef25d3655eef2f83289b60306458ad5e4c17b9d4658bd3` |
| [研究计划模板 Research Plan Template.md](<研究计划模板 Research Plan Template.md>) | 9309 | `b53a7250fd57f754a57a2ac3ddb918eefe7a0ac7a4c28096f39c858486370f44` |
| [文献调研规范 Literature Review Standard.md](<文献调研规范 Literature Review Standard.md>) | 34809 | `2605cf562aee59b2023ecb0954c3639658958c1874ddce79a623dd3df63489a8` |
| [Research Plan Standard — English](research_plan_standard_en.md) | 译本 | 对应正本第一至八章，含原作者参考依据与核验范围 |
| [Research Plan Template — English](research_plan_template_en.md) | 译本 | 对应正本第一至十章，含基线/消融/执行表与变更记录 |

来源：用户指定 DEM 原路径 `文档/规范/研发/研究/研究计划/`，本次实际读取监督助手复制到本仓库 `_staging_from_DEM/` 的文件；没有直接访问 DEM Office 源路径。暂存中的中文名与 ASCII 别名逐字节相同。暂存保留作对照，不是正式文档入口，不纳入 Git。根.gitattributes对这两份正本及本轮文献调研规范设置-text，禁止Git自动换行转换；Git blob与工作区都已逐字节核对，不改变其他文件策略。以后更新先核对新交付版本与差异，不从网址自动替换正本。

## 二、在 RPipe 使用

从[研究工作要求](../../development/research.md)进入；本项目具体方案见[中文研究计划](<../研究计划 RPipe.md>)、[完整英译](<../Research Plan RPipe.md>)，检索与阅读记录见[文献证据](../literature.md)。宏观问题、方案与全部 Study 协议统一在上述项目计划的独立章节，第十一章保留历史全文和模板；不再创建独立 PLAN.md。正式数字、图和结论仍留在相应 Study。模板中建议的通用 `docs/research-plan.md` 不是 RPipe 新增的强制路径，也不要求复制重复计划。

按问题需要合并/简写章节；未知项保留待确认。基线复现优先官方实现、关注代表性主要结果，如实区分权重评测与训练/方法复现。不因模板未填满机械停下，不因复制规范自动授权训练、发布、预算或系统改动。本次没有重写任何已完成 Study 的计划/结果。

## 三、原文依赖与翻译边界

原文为了“原样”保留以下 DEM 内部相对引用；**原 DEM 目录层级不在本仓库，原相对链接不能直接打开**：规范中 `../文献调研/文献调研规范 Literature Review Standard.md`、`../../开发/通用调研规范 General Research Standard.md`，以及第五层上级的校准文献调研大纲。本轮文献规范已按原字节单独落在[同目录文件](<文献调研规范 Literature Review Standard.md>)，供正常导航；没有改写计划正本中的旧相对链接。文献规范自身引用的论文库治理文件尚未迁入。两个同目录研究计划中文正本的互链在本仓库有效。未伪造缺失原件、建立指向无关文档的替代链接或静默改写源文；需要这些依赖时另按授权取得。

英文版将这三类外部集合依赖写成文字说明并指回本页；其他论文/skill 链接保留，不声称本次读过其正文或安装过 skill。原文的2026-10-11 GitHub核验、2026-10-14论文核验属于原作者记录；本次仅完整阅读/复制/翻译用户交付文本，不重新验证二级论文、云端修订或所有外链。

## 四、独立探索议题

以下仅为独立调研，不与模板落库绑定为实施条件，也不修改默认路径或 logger：

- [项目表层 shared 与跨项目不可变输入共享](../../issue/cross_project_shared.md)。
- [实验监控与 logger 选型](../../issue/experiment_monitoring.md)。

## 五、2026-10-16 文献规范补充

完整阅读可读本机 DEM `文档/规范/研发/研究/文献调研规范 Literature Review Standard.md` 的 623 行，以原文件名和原字节落库，字节数/哈希见上表。没有声称通过 Office 读取；没有复制通用调研规范、论文库治理或其他未读集合。原件提供通用要求；本轮实际检索与文献阅读另外登记在 [literature.md](../literature.md)，不把原作者引用自动当本轮已核验来源。中文研究计划和完整英译采用模板十章，不改历史 Study 方案或运行布局。
