# Study Report: my_study

> Plan: [PLAN.md](PLAN.md)
> Date:
> Recipe:
> Location: Study `docs/`。数字读 `../process.json`。

## 1. 怎么跑的（§3）

```bash
python -m rpipe make studies/<name> --num-gpus 1 --init-gpu 0
python -m rpipe launch studies/<name> --num-gpus 1 --init-gpu 0
```

make 打印的 `pack N waits:` （贴实际输出）。error / resume：（有则记 `run_id`）。

## 2. Conclusion

（按 Experiment 聚合，不要扁平 run 列表。）

## 3. Learning curves

![learning curves](./figures/learning_curves.png)

## 4. Runs

| factors | seed | id | metrics |
|---------|------|----|---------|

## 5. Reproduce

同上 make / launch。已 succeeded 的默认跳过。
