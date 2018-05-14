model=$1
dataset="svhn"
gpu=$2
se=$3

time=$(date +"%Y%m%d_%H_%M")
description=$4
post=$5
directory="checkpoints/${model}${description}_${dataset}_${post}_${time}_in_progress"
datadir=".${gpu}"

mkdir -p ${directory}/model
cp $0 ${directory}
python main_sequence.py -a $model -b 128 --lr 0.01 --wd 0.0005 --epochs 240 --save $directory --dataset $dataset --se-reduce $se --gpu $gpu /home/coldmoon/Datasets/SVHN/LMDB${datadir} | tee ${directory}/log.txt

mv ${directory} checkpoints/${model}${description}_${dataset}_${post}_${time}
