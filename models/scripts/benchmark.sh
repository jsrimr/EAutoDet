#!/bin/sh
gpu=( $@ )
gpu_num=$#
echo $gpu_num

data=coco.yaml
ID=20210519-020600
project=runs/train-$ID
LOG_DIR=eval_logs

BATCHSIZE=1

CUDA_VISIBLE_DEVICES=${gpu[0]} python test.py --data $data --batch-size 1 --img 640 512 416 --conf 0.001 --iou 0.65 --weights $project/full_train/weights/stripped_best.pt --project $project --name benchmark --task speed > $LOG_DIR/$ID-speed.log 2>&1 &

