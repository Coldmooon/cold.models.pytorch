model=$1
depth=$2
dataset="imagenet"
se=$3

time=$(date +"%Y%m%d_%H_%M")
description=$4
post=$5
gpu=$6
directory="checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}_in_progress"
datadir=".${gpu}"

mkdir -p ${directory}/model
cp $0 ${1}.py main.py ${directory}

echo "python -u main.py -a $model --depth $depth -b 128 --lr 0.1 --wd 0.0001 --epochs 100 --save $directory --resume $directory/checkpoint.pth.tar --dataset $dataset --workers 8 --se-reduce $se --gpu $gpu /home2/liyang/ILSVRC2012 | tee ${directory}/log.txt" > ${directory}/resume.sh

chmod 777 ${directory}/resume.sh

python -u main.py -a $model --depth $depth -b 128 --lr 0.1 --wd 0.0001 --epochs 100 --save $directory --dataset $dataset --workers 8 --se-reduce $se --gpu $gpu /home2/liyang/ILSVRC2012 | tee ${directory}/log.txt

mv ${directory} checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}
