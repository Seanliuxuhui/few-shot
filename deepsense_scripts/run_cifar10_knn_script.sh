bsub -oo cifar10_knn_10.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_knn_10.sh

bsub -oo cifar10_knn_50.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_knn_50.sh

bsub -oo cifar10_knn_100.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_knn_100.sh

bsub -oo cifar10_knn_500.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_knn_500.sh