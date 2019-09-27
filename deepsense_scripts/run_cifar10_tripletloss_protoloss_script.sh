bsub -oo cifar10_tripletloss_protoloss_10.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_tripletloss_protoloss_10.sh

bsub -oo cifar10_tripletloss_protoloss_50.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_tripletloss_protoloss_50.sh

bsub -oo cifar10_tripletloss_protoloss_100.out -n 4 -M 16000 -W 48:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_tripletloss_protoloss_100.sh

bsub -oo cifar10_tripletloss_protoloss_500.out -n 4 -M 16000 -W 72:0 -R "span[hosts=1] rusage[mem=16000]" -gpu "num=2:mode=shared:mps=yes:j_exclusive=yes" bash cifar10_tripletloss_protoloss_500.sh

