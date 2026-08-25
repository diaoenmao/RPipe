# Study Report: mnist_train_size

> Plan: [PLAN.md](PLAN.md)
> Date: 2026-08-22
> Recipe: mnist_linear (real MNIST + linear)
> Location: Study docs/ (human report).

## 1. Conclusion

With fixed 2 epoch / SGD lr=0.1 / seed=0, larger train_size raises test accuracy: 500 -> 2000 -> 8000 gives **0.764 -> 0.850 -> 0.893**.

## 2. Runs

| train_size | tags | Run id | status | test accuracy | last-batch loss |
|------------|------|--------|--------|---------------|-----------------|
| 500 | baseline | 03d885fff93ebdb8 | succeeded | 0.7642 | 0.415 |
| 2000 | - | 05d8fd2bd9e28389 | succeeded | 0.8495 | 0.384 |
| 8000 | - | 219dd92975be42ed | succeeded | 0.8928 | 0.275 |

Delta vs baseline 500: 2000 +0.0853; 8000 +0.1286.

Layout: ../index.json ; ../runs/<id>/result.json ; ../shared/data/

## 3. Reproduce

`
python -m rpipe run studies/mnist_train_size
`

