# Bugs

> 已知缺陷都留在这里。编号只增不复用。修完把状态改成 `done`，写上方案和验证，不删条目。权威仍是 [CONCEPT.md](CONCEPT.md) → [LAYOUT.md](LAYOUT.md) → 代码。

新增一条用下面的块，**一条一事**。

```text
## B-NNN 短标题
- 状态：open | investigating | wontfix | done
- 看见：怎么复现、期望 vs 实际
- 落点：源文件 / Study / 测试
- 方案：
- 验证：
```

---

## 开放

## B-007 导入 torchvision 时偶发 `torchvision::nms` 不存在
- 状态：open
- 看见：2026-09-26 `launch studies/cifar_grid` 时，mlp 的 eval `f487ffe68a52f01c` 在 `from torchvision import datasets` 处失败：`RuntimeError: operator torchvision::nms does not exist`。同一进程稍后 `retry` 再跑，prepare 通过，eval 成功。`run.log` 里先有这次 ERROR，后面是 `flow succeeded`。其余 7 条没有这个错误。
- 落点：`structure/system/torchvision_load.py`，由 `structure/data/factory.py` 的 `_build_torch_vision` 调用。现场日志 `studies/cifar_grid/runs/f487ffe68a52f01c/assets/logs/run.log`。
- 方案：公开记录（[vision#9174](https://github.com/pytorch/vision/issues/9174)、[vision#8101](https://github.com/pytorch/vision/issues/8101)、[PyTorch 论坛](https://discuss.pytorch.org/t/runtimeerror-operator-torchvision-nms-does-not-exist/192829)）把这句解释成 `torchvision/_C` 没注册上。常见原因是 `torch` 与 `torchvision` 版本不配，或一个是 CPU 轮子、一个是 CUDA 轮子。`extension.py` 会吞掉真正的 `OSError`，然后 `_meta_registrations.py` 给 `nms` 登记 fake 才抛出这句。那边的修法是卸掉后从同一 index 重装配对，不是在同进程里重试导入。本机现在是 `torch 2.11.0+cu128` 与 `torchvision 0.26.0+cu128`，都在 `D:\anaconda3\Lib\site-packages`，`_has_ops()` 为 True。代码里仍会在这句错误上清掉半导入再试最多 3 次，并设 `TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS=1`。这层重试没有公开案例说明能治版本不配；版本不配时每次导入都会失败。
- 验证：单测只把 `import_module` 换成假的，证明重试分支会再调一次。2026-09-26 另开 24 个新进程走 `import_torchvision()`，24 次都成功，这句错误没有再出现。同进程里 `_C` 真失败之后清掉 `sys.modules` 能否恢复，还没有打出来过。

---

## 不修

## B-003 HF Trainer 与 native 在 train_size=8000 的 last accuracy 差约 0.0008
- 状态：wontfix
- 看见：`studies/mnist_native_vs_hf`（2026-09-16）。500 / 2000 三颗 seed 的 accuracy、best_accuracy、train_loss 逐点相同。8000：HF last accuracy mean 0.9069，native 0.9061（差约 +0.0008），和 seed 标准差同量级。
- 落点：`structure/algorithm` 里 `transformers_trainer` 的 cosine / `_EpochScheduler`。该 Study 的 `docs/PLAN.md`。
- 方案：不改 native 去追这个差。500 / 2000 若开始对不齐，再把本条改回 open。
- 验证：当时的 `studies/mnist_native_vs_hf/docs/STUDY_REPORT.md`。

---

## 已完成

## B-001 Windows OpenMP 双份运行时，make/launch 直接退出
- 状态：done
- 看见：本机 conda + torch 上 `python -m rpipe make` 报 `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`，进程退出码非 0。设 `KMP_DUPLICATE_LIB_OK=TRUE` 后能跑完。
- 落点：`src/rpipe/__init__.py`。`docs/STUDY_GUIDE.md` 的 OpenMP 一段。
- 方案：Windows 上 import `rpipe` 时先 import NumPy，让 Conda 与 pip 的两份 `libiomp5md.dll` 共用先加载的那份。不把 `KMP_DUPLICATE_LIB_OK` 写进库。环境仍然双份时，STUDY_GUIDE 要求用 `where.exe libiomp5md.dll` 查来源，并在同一个包管理器里重装，或换干净虚拟环境。
- 验证：`975940a`。2026-09-25 本机不设 `KMP_DUPLICATE_LIB_OK`，`import rpipe` 后再 `import torch` 退出码 0（torch 2.11.0+cu128，cuda True）。

## B-002 `system.device: cpu` 时 make 仍按 GPU 装箱并打印 GPU 行
- 状态：done
- 看见：`studies/mnist_native_vs_hf` 基底是 `device: cpu`，`make --num-gpus 1 --init-gpu 0` 仍打印 `GPU0 NVIDIA … usable …GiB` 和 wait 组。launch 日志也是 `+ gpu=0 <run_id>`。
- 落点：`structure/make/capacity.py`、`structure/make/schedule.py`。
- 方案：按 Run 的 `system.device` 分队列。`cpu` 不探测 GPU、不设 `CUDA_VISIBLE_DEVICES`、装箱时去掉 `gpu`，摘要写 `CPU`。`cuda` 才按显存装箱。混合 Study 里两类分开成组。
- 验证：`975940a`。`tests/rpipe/structure/make/test_capacity.py`、`tests/rpipe/structure/make/test_make.py`。

## B-004 单测里 `lr_scheduler.step()` 早于 `optimizer.step()` 的 PyTorch 警告
- 状态：done
- 看见：`python tests/run.py --core` 时 `test_optim.py` / `test_train.py` 出现 `UserWarning`。用例仍通过。
- 落点：`tests/rpipe/structure/algorithm/test_optim.py`、`test_train.py`。生产循环在 `structure/algorithm/train/__init__.py`：先 `optimizer.step()`，再 `scheduler.step()`。
- 方案：警告来自测试为了读 lr 单独调用 `sched.step()`。两处测试都先 `opt.step()` 再 `sched.step()`。生产循环顺序保持 PyTorch 1.1+ 的约定，没有改。
- 验证：`975940a`。

## B-005 make 下载共享数据时终端无输出
- 状态：done
- 看见：`python -m rpipe make studies/cifar_grid` 在展开 index 之后下载 CIFAR，tqdm 被关掉，`shared data:` 和 jobs 路径都要等整段 make 结束才打印。终端几分钟空白，分不清是在下载还是卡住。期望：进行中能看见当前阶段。
- 落点：`structure/artifact/activity.py`、`flow/cli.py`、`structure/data/prepare.py`
- 方案：make 每个阶段立刻 flush 一行，并写入 study 根上的 `activity.json`（不进 Git）：`make: expand`、`make: shared <name> download <origin> <url>`、`ready`、`cached`、`make: pack`。另开终端 `rpipe status` 时，有这份文件就先打这一行。make 成功后删除 `activity.json`。下载仍然不刷 tqdm。
- 验证：`tests/rpipe/structure/make/test_make.py` 的 CPU make 打出 `make: expand` 和 `make: pack`，结束后没有 `activity.json`。`tests/rpipe/structure/data/test_prepare.py` 在 `DataFactory.build` 返回前已经打出 download 行。`tests/rpipe/flow/test_status.py` 在没有 index 时打出活动行。2026-09-26 重跑 `make studies/cifar_grid` 时，终端在下载开始前就打出了 `make: origin`、`make: expand` 和 `make: shared CIFAR10 download`。

## B-006 国内下载 torchvision 数据集走不通
- 状态：done
- 看见：`studies/cifar_grid` 的 CIFAR10 经 `torchvision.datasets.CIFAR10(download=True)` 拉多伦多 `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`（约 170MB）。本机 00:31–00:53 约 20 分钟只到约 80MB。中断后 `shared/data/cifar10/` 留下半截 tar.gz；当时目录非空会被当成缓存跳过。期望：Study 选定国内或国外源，半截包不当成已就绪，国内源能下完。
- 落点：`structure/origin.py`。`study.yaml` 顶层 `origin`。`studies/cifar_grid` 为 `domestic`。
- 方案：Study 在 `study.yaml` 顶层写一次 `origin: foreign` 或 `domestic`（缺省 `foreign`，且不改已有的 `HF_ENDPOINT`）。数据和模型都读这一项，不写在 `data` 下面。`foreign`：torchvision 官方地址，模型 hub `https://huggingface.co`。`domestic`：数据 `https://dataset.bj.bcebos.com`（CIFAR10 / CIFAR100 / MNIST），模型 hub `https://hf-mirror.com`（写入 `HF_ENDPOINT`）。没有国内镜像的数据集（SVHN、FashionMNIST）选 `domestic` 会直接报错。成功后才写 `shared/data/<root>/.ready`，且标记里的 origin 要一致才跳过。md5 不对的 tar 在下载前删掉。阶段行先打 `make: origin … model …`。
- 验证：删掉半截目录后，2026-09-26 01:20 的 `make studies/cifar_grid` 约 50 秒结束，退出码 0。tar 为 170498071 字节，已解压 `cifar-10-batches-py`，`.ready` 内容是 `domestic`。8 条 job 已装箱。
