# v0.3.0 长矩阵中断后的接续方案

## 一、状态与边界

2026-10-13 20:59左右，原生launch及其观察进程退出；没有取得Flow错误定稿或进程退出码，原因未确定。30条train已正式succeeded并完成80000步，CIFAR10 / ResNet18 seeds2、3停在约24400步，32条eval未执行。观察JSON的running字段已过期，不能据此认为程序仍在执行。

软件仍为最新已核对main / v0.3.0的804b3823。库与Study计算源码、声明、数据、FP32、CPU2、batch250、完整test、80000步、eval200 / 400点及4.5GiB预算保持不变；历史自定义sampler拒绝中断续训的合同保持不变。

## 二、原生接续清单

- 先保全原index、provenance、jobs、观察记录与前置门字节，并记录中断Run的日志 / checkpoint位置；没有错误定稿的原Run不补造成功或Flow异常。
- 用现有curves.inspect_run逐条核对30条完整训练：连续标记、400点、完整数据、最新80000步、best与全程最大值及原结果 / 配置 / provenance。原Run和原来源记录保留，不重新写成新来源。
- 仅为中断的两个seed的train及各自eval分配 `main-exp-v030-20261013-recovery01`。使用库make的write_run_configs / write_study_index / write_provenance构建接续清单，保留原30条已完成train及其30条原eval配置，替换4个未完成槽位；每个科学组合和seed仍恰好一个train和一个eval。
- 旧 / 新provenance的源码files映射必须完全相等，64个配置仅4个Run的身份version改变，其他科学字段一致。新来源绑定新清单，旧来源绑定旧清单；没有回填旧成功的provenance，也不把原30条称为新执行。
- 按原生plan_jobs跳过核验成功的30条，使用原装箱和保护池生成34个待执行任务：2条新连续80000步train及32条own-best eval。仍保留train→eval屏障、唯一Run写入者和实际资源准入。
- 原生CLI `rpipe launch`执行接续；终验读取接续index。正式报告明确两次调度和逐Run实际version / provenance；长矩阵不称一次不间断launch，每条成功train须各自连续80000步。

这是同一科学基线的中断处理，没有源码修复或精度变化。若来源不一致、原30条完整性不通过或资源准入失败，停止该接续方案，保留证据并重新确定受影响范围，不混用不同科学基线。

## 三、正式证据

接续的逐槽替换、原30条完整性与source映射、资源准入、原始快照清单写入v030_recovery_plan.json；执行使用新观察JSON / 日志，原观察和原中断资产保留。最终核对index / process / COMPARISON与全部32个best来源及固定曲线门。临时编排脚本、启动日志和资源JSONL在.tmp，仅本机；计算和调度仍由现有库与Study入口执行。
