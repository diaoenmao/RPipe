#!/bin/bash

python make.py --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

