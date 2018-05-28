# python main.py -a resnet20 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee ResNet20_3_lr0.1_bs128_epoch200_GTX1080.txt 
# python main.py -a resnet20 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee senet20_2_lr0.1_bs128_epoch200_GTX1080.txt 
# python main.py -a resnet110 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee senet110_1_lr0.1_bs128_epoch200_GTX1080.txt 
# python main.py -a resnet110 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee senet110_2_lr0.1_bs128_epoch200_GTX1080.txt 
# python main.py -a resnet56 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee senet56_1_lr0.1_bs128_epoch200_GTX1080.txt 
# python main.py -a resnet56 -b 128 --epochs 200  /media/coldmoon/ExtremePro960G/ILSVRC2012/ --gpu 0 | tee senet56_2_lr0.1_bs128_epoch200_GTX1080.txt 

# python main_sequence.py -a stm11 -b 128 --lr 0.01 --wd 0.0005 --epochs 240 --resume /home/coldmoon/Developer/cold.models.pytorch/checkpoint.pth.tar --dataset svhn --se-reduce 16 --gpu 0 /home/coldmoon/Datasets/SVHN/LMDB | tee STMulti11_1_bn_gauss_nolengthlabel_lr0.01_bs128_epoch240_svhn_GTX1080.txt


# ./train.sh stplinconv 0 16 _fcn_BNorm_avgbeforest_nost1_nobn_nodropout_gauss_ _GTX1080






# ./cifar.sh aresnet20 0 16 _ALU_r4_r8_r16_BN_msra_bs128_ _cifar10_GTX1080
# ./cifar.sh aresnet32 0 16 _ALU_r4_r8_r16_BN_msra_bs128_ _cifar10_GTX1080
# ./cifar.sh aresnet44 0 16 _ALU_r4_r8_r16_BN_msra_bs128_ _cifar10_GTX1080
# ./cifar.sh aresnet56 0 16 _ALU_r4_r8_r16_BN_msra_bs128_ _cifar10_GTX1080
# ./cifar.sh aresnet110 0 16 _ALU_r4_r8_r16_BN_msra_bs128_ _cifar10_GTX1080


./mnist.sh transerror 0 16 _STN7_nobn_gauss_lr0.01_bs64_ _conv1aug360_mnist-rot-12k_GTX1080
