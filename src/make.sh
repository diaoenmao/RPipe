#!/bin/bash

modes=('base')
resume_mode=0
num_gpus=1
round=8
num_experiments=2

for mode in ${modes[@]}; do
    python make.py --mode $mode --run train --resume_mode $resume_mode --num_gpus $num_gpus --round $round --num_experiments $num_experiments
    python make.py --mode $mode --run test --resume_mode $resume_mode --num_gpus $num_gpus --round $round --num_experiments $num_experiments
  done