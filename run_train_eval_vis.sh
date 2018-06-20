#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python deeplab/train.py --logtostderr --training_number_of_steps=100 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=1 --dataset="mscoco" --tf_initial_checkpoint="./deeplab/datasets/deeplabv3_pascal_trainval/model.ckpt" --train_logdir="./deeplab/logs/train" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"

CUDA_VISIBLE_DEVICES=1 python deeplab/eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="mscoco" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/logs/train/" --eval_logdir="./deeplab/logs/eval" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"


CUDA_VISIBLE_DEVICES=1 python deeplab/vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="mscoco" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/logs/train/" --vis_logdir="./deeplab/logs/vis" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"



