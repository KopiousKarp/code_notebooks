#!/bin/bash
python HyperParametertuning.py \
	--coco_roots /work/2021_annot/images /work/2022_annot/images /work/2023_annot/images /work/2024_annot/images \
	--coco_annFiles /work/2021_annot/2021_annotations.json /work/2022_annot/2022_annotations.json /work/2023_annot/2023_annotations_corrected.json /work/2024_annot/2024_annotations.json \
	--learning_rates .0001 \
	--weight_decays 0.00001 \
	--train_split_ratio 0.8 \
	--epochs 50 \
	--batch_size 16 \
	--num_workers 4 \
	--no_augmentations
