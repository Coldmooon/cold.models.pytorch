model=$1
depth=$2
se=$3

time=$(date +"%Y%m%d_%H_%M")
description=$4
post=$5
gpu=$6
directory="checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}_in_progress"
datadir=".${gpu}"
dataset="cifar10"

mkdir -p ${directory}/model
cp $0 ${directory}

echo "python main.py -a $model -b 128 --lr 0.1 --wd 0.0005 --epochs 200 --save $directory --resume $directory/checkpoint.pth.tar --dataset $dataset --se-reduce $se --gpu $gpu /home/coldmoon/Datasets/SVHN/LMDB${datadir} | tee ${directory}/log.txt" > ${directory}/resume.sh

chmod 777 ${directory}/resume.sh

# -u: https://stackoverflow.com/questions/21662783/linux-tee-is-not-working-with-python
python -u main.py -a $model --depth $depth -b 128 --lr 0.1 --wd 0.0005 --epochs 200 --save $directory --dataset $dataset --se-reduce $se --gpu $gpu --print-freq 2 /home/coldmoon/Datasets/SVHN/LMDB${datadir} | tee ${directory}/log.txt

mv ${directory} checkpoints/${model}${depth}${description}_${dataset}_${post}_${time}
