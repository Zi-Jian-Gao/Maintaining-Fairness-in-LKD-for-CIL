#!/bin/bash

'''Lwf test with CIFAR-100 dataset'''

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "7" --method "normal" --dataset "cifar100" loadpre 0 > normal_lwf_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "6" --method "KL" --dataset "cifar100" loadpre 0 > KL_lwf_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "1" --method "interintra" --dataset "cifar100" loadpre 0 > interintra_Lwf_lambda5_SplitT5.log 2>&1 &


'''Models w/interintra'''

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "0" --method "interintra" --dataset "cifar100" loadpre 0 > inter_intra1_3_lwf_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 10  --increment 10 --device "1" --method "interintra" --dataset "cifar100" loadpre 0 > inter_intralambda1_3_lwf_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "2" --method "interintra" --dataset "cifar100" loadpre 0 > inter_intralambda1_3_lwf_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 5  --device "3" --method "interintra" --dataset "cifar100" loadpre 0 > inter_intralambda1_3_lwf_cifar_HalfT11.log 2>&1 &

# nohup python main.py --config './exps/icarl.json' --init_cls 20  --increment 20 --device "0" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intra1_3_icarl_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 10  --increment 10 --device "1" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_icarl_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 50  --increment 10 --device "2" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_icarl_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 50  --increment 5  --device "3" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_icarl_cifar_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/wa.json' --init_cls 20  --increment 20 --device "0" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intra1_3_wa_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 10  --increment 10 --device "1" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_wa_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 10 --device "2" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_wa_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 5  --device "3" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_wa_cifar_HalfT11.log 2>&1 &

# nohup python main.py --config './exps/wa.json' --init_cls 20  --increment 20 --device "6" --method "inter" --dataset "cifar100" --loadpre 0 > inter_wa_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 10  --increment 10 --device "6" --method "inter" --dataset "cifar100" --loadpre 0 > inter_wa_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 10 --device "7" --method "inter" --dataset "cifar100" --loadpre 0 > inter_wa_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 5  --device "7" --method "inter" --dataset "cifar100" --loadpre 0 > inter_wa_cifar_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/podnet.json' --init_cls 20  --increment 20 --device "4" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intra1_3_podnet_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 10  --increment 10 --device "4" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_podnte_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 50  --increment 10 --device "7" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_podnet_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 50  --increment 5  --device "7" --method "interintra" --dataset "cifar100" --loadpre 0 > inter_intralambda1_3_podnet_cifar_HalfT11.log 2>&1 &

# nohup python main.py --config './exps/podnet.json' --init_cls 20  --increment 20 --device "6" --method "inter" --dataset "cifar100" --loadpre 0 > inter_podnet_cifar_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 10  --increment 10 --device "6" --method "inter" --dataset "cifar100" --loadpre 0 > inter_podnet_cifar_S10.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 50  --increment 10 --device "4" --method "inter" --dataset "cifar100" --loadpre 0 > inter_podnet_cifar_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 50  --increment 5  --device "7" --method "inter" --dataset "cifar100" --loadpre 0 > inter_podnet_cifar_HalfT11.log 2>&1 &




'''
 Models w/FKD
'''

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "2" --method "feature" --dataset "cifar100" loadpre 0 > flat_lwf_cifar_HalfT6.log 2>&1 &

# nohup python main.py --config './exps/icarl.json' --init_cls 50  --increment 10 --device "1" --method "feature" --dataset "cifar100" loadpre 0 > flat_icarl_cifar_HalfT6_epoch170.log 2>&1 &

# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 10 --device "1" --method "feature" --dataset "cifar100" --loadpre 0 > flat_wa_cifar_HalfT6_50.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "3" --method "passfeature" --dataset "cifar100" loadpre 0 > passflat_lwf_cifar_HalfT6.log 2>&1 &

# nohup python main.py --config './exps/icarl.json' --init_cls 50  --increment 10 --device "2" --method "passfeature" --dataset "cifar100" loadpre 0 > passflat_icarl_cifar_HalfT6_epoch170.log 2>&1 &

# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 10 --device "2" --method "passfeature" --dataset "cifar100" --loadpre 0 > passflat_wa_cifar_HalfT6_50.log 2>&1 &




'''
Supplementary ablation experiment
'''

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "0""1" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=2 > lwf_lambd2_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "1""0" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=3 > lwf_lambd3_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "2""3" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0.5 > lwf_lambd1_2_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "4""3" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0.3333 > lwf_lambd1_3_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "5""6" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=1 > lwf_lambd1_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "6""5" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=-1 > lwf_lambd_inter_SplitT5.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "7""8" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0 > lwf_lambd_intra_SplitT5.log 2>&1 &



# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "6""5" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=2 > lwf_lambd2_HT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "5""6" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=3 > lwf_lambd3_HT6.log 2>&1 &

# # nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "4""5" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0.5 > lwf_lambd1_2_HT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "1""3" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0.3333 > lwf_lambd1_3_HT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "2""1" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=1 > lwf_lambd1_HT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "1""2" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd=-1 > lwf_lambd_inter_HT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "9""6" --method "interintra" --dataset "imagenet100" loadpre 0 --lambd 0 > lwf_lambd_intra_HT6.log 2>&1 &



'''
 Models w/RKD
'''
# nohup python main.py --config './exps/icarl.json' --init_cls 50  --increment 10 --device "9" --method "rkd" --dataset "cifar100" loadpre 0 > icarl_test_HalfT6.log 2>&1 &

# nohup python main.py --config './exps/wa.json' --init_cls 50  --increment 10 --device "9" --method "rkd" --dataset "cifar100" loadpre 0 --lambd 1 > wa_test_HalfT6.log 2>&1 &

# nohup python main.py --config './exps/lwf.json' --init_cls 50  --increment 10 --device "1" --method "rkd" --dataset "cifar100" loadpre 0 --lambd 1 > lwf_test_HalfT6.log 2>&1 &



'''Tiny-ImageNet dataset test'''

# nohup python main.py --config './exps/lwf.json' --init_cls 100  --increment 20 --device "0" --method "normal" --dataset "tiny" --loadpre 0 > normal_lwf_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 100  --increment 10 --device "1" --method "normal" --dataset "tiny" --loadpre 0 > normal_lwf_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/icarl.json' --init_cls 100  --increment 20 --device "2" --method normal --dataset "tiny" --loadpre 0 > normal_icarl_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 100  --increment 10 --device "5" --method normal --dataset "tiny" --loadpre 0 > normal_icarl_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/wa.json' --init_cls 100  --increment 20 --device "7" --method normal --dataset "tiny" --loadpre 0 > normal_wa_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/wa.json' --init_cls 100  --increment 10 --device "7" --method normal --dataset "tiny" --loadpre 0 > normal_wa_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/replay.json' --init_cls 40  --increment 40 --device "5" --method normal --dataset "tiny" --loadpre 0 > normal_replay_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/replay.json' --init_cls 20  --increment 20 --device "5" --method normal --dataset "tiny" --loadpre 0 > normal_replay_tiny_SplitT10.log 2>&1 &


# nohup python main.py --config './exps/podnet.json' --init_cls 40  --increment 40 --device "6" --method normal --dataset "tiny" --loadpre 0 > normal_podnet_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 20  --increment 20 --device "7" --method normal --dataset "tiny" --loadpre 0 > normal_podnet_tiny_SplitT10.log 2>&1 &


# nohup python main.py --config './exps/replay.json' --init_cls 40  --increment 40 --device "3" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intra1_3_replay_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/replay.json' --init_cls 20  --increment 20 --device "3" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_replay_tiny_S10.log 2>&1 &
# nohup python main.py --config './exps/replay.json' --init_cls 100  --increment 20 --device "4" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_replay_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/replay.json' --init_cls 100  --increment 10 --device "4" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_replay_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/icarl.json' --init_cls 40  --increment 40 --device "0" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intra1_3_icarl_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 20  --increment 20 --device "1" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_icarl_tiny_S10.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 100  --increment 20 --device "2" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_icarl_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/icarl.json' --init_cls 100  --increment 10 --device "2" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_icarl_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/lwf.json' --init_cls 40  --increment 40 --device "0" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intra1_3_lwf_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 20  --increment 20 --device "1" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_lwf_tiny_S10.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 100  --increment 20 --device "2" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_lwf_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/lwf.json' --init_cls 100  --increment 10  --device "3" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_lwf_tiny_HalfT11.log 2>&1 &


# nohup python main.py --config './exps/podnet.json' --init_cls 40  --increment 40 --device "4" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intra1_3_podnet_tiny_SplitT5.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 20  --increment 20 --device "2" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_podnet_tiny_S10.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 100  --increment 20 --device "6" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_podnet_tiny_HalfT6.log 2>&1 &
# nohup python main.py --config './exps/podnet.json' --init_cls 100  --increment 10  --device "7" --method "interintra" --dataset "tiny" --loadpre 0 > inter_intralambda1_3_podnet_tiny_HalfT11.log 2>&1 &





