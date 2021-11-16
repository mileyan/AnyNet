python3 finetune.py --maxdisp 192 --with_spn --datapath /home/bart/datasets/kitti2015/training/ \
    --save_path results/kitti2015 --datatype 2015 --pretrained checkpoint/kitti2015_ck/checkpoint.tar \
    --split_file checkpoint/kitti2015_ck/split.txt --evaluate

