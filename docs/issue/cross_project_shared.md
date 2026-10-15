# 项目表层 shared 的跨项目共享调研

独立探索 Issue：[89](https://github.com/DreamSoul-AI/RPipe/issues/89)。

## 一、结论与状态

记录日期2026-10-16，**探索议题，不是已确认缺陷或实施决定**。项目表层 `shared/{data,model}` 继续作为各 Study 共用不可变输入的逻辑入口，默认位置不变。跨项目可以先评估现有 `shared.root` 显式指向同一资产根，或各项目的 `shared` symlink 指向同一物理根；两种方式可分别使用，不与 logger/研究模板绑在一起。本次未建链接、迁移数据、改配置/运行时或部署跨项目共享。

Enmao 提供的“以前用 symlink”是需求背景；尚未取得旧项目链接清单、原部署/权限或磁盘证据，不能写成已验证本机拓扑。本文只读源码及官方文件系统文档，不做吞吐或速度比较。

## 二、现有代码与两种候选

依据[layout](../code/layout.md#跨-study-共享资产2026-10-14)和[shared.py](../../src/rpipe/structure/artifact/shared.py)的 `shared_paths`：`studies/<study>` 默认解析到项目根 shared，非标准 Study 布局使用其父目录；可显式配置 `shared.root`，`data/model: study` 可逐类私有。`cache_lock` 解析物理根后加 OS 文件锁，`prepared_resource` 依据声明清单/manifest发布。它们不是一个全局跨机器资产管理器。

| **方案** | **逻辑入口与变化** | **优点** | **必须核对的边界** |
| --- | --- | --- | --- |
| 现有显式 shared.root | Study 声明显式资产根；默认仍项目/shared | 不依赖新目录链接，不需新增配置字段 | 配置是机器相关还是可迁移；两项目是否真指同一版本/格式；显式相对路径按 Study 解析 |
| 项目 shared 是 symlink | 项目仍用 shared/data、shared/model，物理资产根可在项目外 | 保留表层入口，契合既有使用习惯 | 链接创建权限、目标持久性、重定位、断链与同物理根识别；不能按链接外观判断缓存相同 |
| 复制独立缓存 | 仍用现有默认根 | 隔离简单，无跨项目并发写 | 重复容量；仅有真实不兼容才私有，不将它当默认迁移目标 |

Python `Path.resolve()` 处理符号链接，而路径归一化可能改变含链接路径的语义；`symlink_to` 中接收者为链接本身。链接管理与字符串拼接不能混同。[Python3.13 pathlib](https://docs.python.org/3.13/library/pathlib.html#pathlib.Path.resolve)

Windows 支持指向文件/目录的符号链接；创建权限及开发者模式/非特权标志是部署条件，不能假设每台机器均可无特权创建。这里不修改系统开关，也不将 junction、网络共享和 symlink 当作相同承诺。[Microsoft Symbolic Links](https://learn.microsoft.com/en-us/windows/win32/fileio/symbolic-links)、[CreateSymbolicLinkW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createsymboliclinkw)

## 三、独立性与正确性合同

1. 共享对象仅是固定版本原始数据、不可变派生输入、固定 revision 权重/tokenizer。Dataset/模型对象、checkpoint、sampler/RNG、tracker和报告留在各 Run；不得以共享目录实现可变对象复用。
2. 同名资源需按实际版本/格式隔离，不只比较目录名。旧 manifest、真实输入摘要、许可/来源和读取方兼容性须可核对；不把 Git 源码 freeze 自动当作数据版本证明。
3. 两项目同时准备同一根应锁住同一物理资源，builder只构建一次，失败不得写ready或覆盖旧输入；原子发布不替代构建锁。当前已有本机同根并发测试，不等于已覆盖任意两项目、UNC/NFS/SMB或跨机器锁。
4. 执行 `rpipe shared <study> --migrate` 前仍按现有冲突校验和显式迁移合同，保留旧链接/字节。调研不自动执行命令、不删除输入或追随链接递归清理外部目标。

## 四、若将来实施，先做的正确性验证

在临时目录创建两个最小项目，共同指向一个合成资产根；分开验证显式root与symlink（平台不允许链接时明示skip）。检查相同实际根、单次构建、同manifest/输入、锁释放、第二进程重用；注入失败后旧证据与peer保全。再覆盖一类私有输入、一类共享输入、断链、目标拒读和不同版本冲突。只用小合成输入与合同断言，不测速度、不跑GPU、不动正式数据。

通过这些合同后再决定是否仅补接入文档，或确有现有代码不能满足的缺口；目前证据不足以新增全局缓存服务、目录层级、权限管理器或新CLI。后续工程Issue需先有真实失败回归。
