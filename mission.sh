#!/bin/bash

range='^[0-9]+$'
if ! [[ $1 =~ $range ]] ; then
   echo "ERROR: please give a GPU ID (0-7)" >&2; exit 1
fi

gpu=$1

./scripts/cifar.sh resnet 44 16 _ _VCLab_1080Ti${gpu}_ $gpu
./scripts/cifar.sh resnet 32 16 _ _VCLab_1080Ti${gpu}_ $gpu


# ./mnist.sh transerror 0 16 _STN7_nobn_gauss_lr0.01_bs64_ _conv1aug360_mnist-rot-12k_GTX1080
# ./imagenet.sh senet 50 16 _addBN1d_in_SE_lr0.1_wd1e-4_bs128_ _imagenet_DNN 0,1,2,3
