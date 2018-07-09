#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python deeplab/train.py --logtostderr --training_number_of_steps=100 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=1 --dataset="mscoco" --tf_initial_checkpoint="./deeplab/datasets/deeplabv3_pascal_trainval/model.ckpt" --train_logdir="./deeplab/logs/train" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"

CUDA_VISIBLE_DEVICES=1 python deeplab/eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="mscoco" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/logs/train/" --eval_logdir="./deeplab/logs/eval" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"


CUDA_VISIBLE_DEVICES=1 python deeplab/vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="mscoco" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/logs/train/" --vis_logdir="./deeplab/logs/vis" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"


CUDA_VISIBLE_DEVICES=1 python deeplab/export_model.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --output_stride=4 --crop_size=513 --crop_size=513 --dataset="mscoco" --checkpoint_path="./deeplab/datasets/deeplabv3_pascal_trainval/model.ckpt" --export_path="./deeplab/logs/export" --dataset_dir="/home/projectai_segm/Raj/tfrecord/" --num_classes=2

CUDA_VISIBLE_DEVICES=1 python deeplab/eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="mscoco" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/datasets/deeplabv3_pascal_trainval/model.ckpt" --eval_logdir="./deeplab/logs/eval-base" --dataset_dir="/home/projectai_segm/Raj/tfrecord/"




CUDA_VISIBLE_DEVICES=1 python deeplab/export_model.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="mscoco" --checkpoint_path="./deeplab/logs/train/model.ckpt-80000" --export_path="./deeplab/logs/export/frozen_inference_graph.pb" --dataset_dir="/home/projectai_segm/Raj/tfrecord_MSCOCO/" --num_classes=2

CUDA_VISIBLE_DEVICES=1 python deeplab/vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=911 --vis_crop_size=1153 --dataset="humandavissf" --checkpoint_dir="/home/projectai_segm/Raj/sassy_semicolon/deeplab/logs/train_humanDAVISsf/" --vis_logdir="./deeplab/logs/vis_humanDAVISsf" --dataset_dir="/home/projectai_segm/Raj/tfrecord_humanDAVISsf/"

CUDA_VISIBLE_DEVICES=1 python deeplab/export_model.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=911 --vis_crop_size=1153 --crop_size=911 --crop_size=1153 --dataset="humandavissf" --checkpoint_path="./deeplab/logs/train_blurredhumanDAVIS/model.ckpt-15000" --export_path="./deeplab/logs/export/frozen_inference_graph.pb" --num_classes=2
