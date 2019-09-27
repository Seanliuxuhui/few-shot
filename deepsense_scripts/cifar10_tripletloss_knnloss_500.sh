source activate py36
conda activate xuhui
source /opt/DL/cudnn/bin/cudnn-activate
source /opt/DL/pytorch/bin/pytorch-activate
source /opt/DL/tensorflow/bin/tensorflow-activate
python -m few_shot.train_knn_classifier --n-train-samples 400 --k-train-classes 10 --q-train-samples 100 --n-test-samples 500 --k-test-classes 10 --q-test-samples 0 --n-train 5 --k-train 10  --k-test 10 --dataset cifar10 --epochs 3000 --use-resnet-model True --model-name cifar10_tripletloss_500.tar --checkpoint-path cifar10_tripletloss_knnloss_500.tar --tensorboard-logdir cifar10_tripletloss_knnloss_500 --channels 3 --z-dim 64  --use-custom-resnet-model True --train True --need-evaluate True