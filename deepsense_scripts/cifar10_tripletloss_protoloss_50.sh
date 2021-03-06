source activate py36
conda activate xuhui
source /opt/DL/cudnn/bin/cudnn-activate
source /opt/DL/pytorch/bin/pytorch-activate
source /opt/DL/tensorflow/bin/tensorflow-activate
python -m few_shot.adapt_protoloss --n-train-samples 40 --k-train-classes 10 --q-train-samples 10 --n-test-samples 500 --k-test-classes 10 --q-test-samples 0 --n-train 5 --k-train 10 --q-train 5 --k-test 10  --dataset cifar10 --epochs 2000 --use-resnet-model True --model-name cifar10_tripletloss_50.tar --checkpoint-path cifar10_tripletloss_protoloss_50.tar --tensorboard-logdir cifar10_tripletloss_protoloss_50 --channels 3 --z-dim 64  --use-custom-resnet-model True --use-pytorch-pretrain-model False --train True