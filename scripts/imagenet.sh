#!/bin/bash

# https://stackoverflow.com/questions/821396/aborting-a-shell-script-if-any-command-returns-a-non-zero-value
set -e

model=$1
depth=$2
se=$3

time=$(date +"%Y%m%d_%H_%M")
description=$4
post=$5
gpu=$6
datadir=".${gpu}"
dataset=$7
directory="checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}_in_progress"

cp $0 mission.sh main.py models/${model}.py ${directory}

# -u: https://stackoverflow.com/questions/21662783/linux-tee-is-not-working-with-python

echo "python -u main.py -a $model -d $depth -b 256 --lr 0.1 --wd 0.0001 --epochs 100 --se-reduce $se \
     --dist-url 'tcp://127.0.0.1:10000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --gpu $gpu \
     --save $directory --resume $directory/checkpoint.pth.tar \
     -j 32 --print-freq 2 --dataset $dataset \
     /ssd_data/yangli/ImageNet2012 | tee ${directory}/log.txt" \
     > ${directory}/resume.sh

chmod 777 ${directory}/resume.sh

# -u: https://stackoverflow.com/questions/21662783/linux-tee-is-not-working-with-python
python -u main.py -a $model -d $depth -b 256 --lr 0.1 --wd 0.0001 --epochs 100 --se-reduce $se \
       --dist-url 'tcp://127.0.0.1:10000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --gpu $gpu \
       --save $directory \
       -j 32 --print-freq 2 --dataset $dataset \
       /ssd_data/yangli/ImageNet2012 | tee ${directory}/log.txt

mv ${directory} checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}
