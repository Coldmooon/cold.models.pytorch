model=$1
dataset="svhn"
gpu=$2

time=$(date +"%Y%m%d_%H_%M")
description=""
post=""
directory="checkpoints/${model}${description}_${dataset}_${post}_${time}_in_progress"

mkdir -p ${directory}/model
python main_sequence.py -a $model -b 128 --lr 0.01 --wd 0.0005 --epochs 240 --save $directory --dataset $dataset --se-reduce 16 --gpu $gpu /home/coldmoon/Datasets/SVHN/LMDB | tee ${directory}/log.txt

mv ${directory} checkpoints/${model}${description}_${dataset}_${post}_${time}
