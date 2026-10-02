# Numbers: mnist_cnn_budget

Generated from `process.json`. The conclusion stays in `STUDY_REPORT.md`.

## Experiments

| factors | metric | mean | std | min | max | n |
|---|---|---:|---:|---:|---:|---:|
| algorithm.mode=train | accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=train | best_value | 0.0963 | 0.0419 | 0.0608 | 0.1425 | 3 |
| algorithm.mode=train | elapsed_seconds | 17.9801 | 9.5859 | 6.9114 | 23.5630 | 3 |
| algorithm.mode=train | test_accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=train | test_loss | 0.0963 | 0.0419 | 0.0608 | 0.1425 | 3 |
| algorithm.mode=train | train_accuracy | 80.8124 | 7.4055 | 72.8900 | 87.5607 | 3 |
| algorithm.mode=train | train_loss | 0.5926 | 0.2148 | 0.3886 | 0.8167 | 3 |
| algorithm.mode=eval | accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=eval | best_accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=eval | elapsed_seconds | 2.4657 | 0.4013 | 2.0043 | 2.7337 | 3 |
| algorithm.mode=eval | eval_accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=eval | test_accuracy | 96.9300 | 1.2582 | 95.5800 | 98.0700 | 3 |
| algorithm.mode=eval | test_loss | 0.0963 | 0.0419 | 0.0608 | 0.1425 | 3 |

## Runs

| factors | seed | id | metrics | log |
|---|---:|---|---|---|
| algorithm.mode=train | 0 | `cb3e0009eae4d66b` | accuracy=95.5800, best_value=0.1425, elapsed_seconds=23.5630, test_accuracy=95.5800, test_loss=0.1425, train_accuracy=72.8900, train_loss=0.8167 | [run.log](../runs/cb3e0009eae4d66b/assets/logs/run.log) |
| algorithm.mode=train | 1 | `7a73df1b02869b59` | accuracy=98.0700, best_value=0.0608, elapsed_seconds=23.4659, test_accuracy=98.0700, test_loss=0.0608, train_accuracy=87.5607, train_loss=0.3886 | [run.log](../runs/7a73df1b02869b59/assets/logs/run.log) |
| algorithm.mode=train | 2 | `032cf3b849f5673a` | accuracy=97.1400, best_value=0.0856, elapsed_seconds=6.9114, test_accuracy=97.1400, test_loss=0.0856, train_accuracy=81.9867, train_loss=0.5726 | [run.log](../runs/032cf3b849f5673a/assets/logs/run.log) |
| algorithm.mode=eval | 0 | `dcb6b5e904073e97` | accuracy=95.5800, best_accuracy=95.5800, elapsed_seconds=2.7337, eval_accuracy=95.5800, test_accuracy=95.5800, test_loss=0.1425 | [run.log](../runs/dcb6b5e904073e97/assets/logs/run.log) |
| algorithm.mode=eval | 1 | `c88be77461e7326c` | accuracy=98.0700, best_accuracy=98.0700, elapsed_seconds=2.6590, eval_accuracy=98.0700, test_accuracy=98.0700, test_loss=0.0608 | [run.log](../runs/c88be77461e7326c/assets/logs/run.log) |
| algorithm.mode=eval | 2 | `d18a81b715f60ab8` | accuracy=97.1400, best_accuracy=97.1400, elapsed_seconds=2.0043, eval_accuracy=97.1400, test_accuracy=97.1400, test_loss=0.0856 | [run.log](../runs/d18a81b715f60ab8/assets/logs/run.log) |
