#!/bin/bash

for fold in {0..4};
do
	echo "Training The Classification Model Now"
	echo "This is Fold ${fold}"
	#train with resnet50
	python -m kuzushiji.object_classification.main \
        _runs/detection_gt.csv \
        --output-dir _runs/classification_fold${fold} \
        --fold ${fold} \
        --base resnet50 

	#validation with test time augmentation
	python -m kuzushiji.object_classification.main \
        _runs/detection_gt.csv \
        --output-dir _runs/classification_fold${fold} \
        --fold ${fold} \
        --base resnet50 \
        --resume _runs/classification_fold${fold}/model.best.p \
        --print-model 0 \
        --n-tta 4 \
        --test-only > _runs/classification_fold${fold}/validation.txt

	#submission
	python -m kuzushiji.object_classification.main \
        _runs/detection_gt.csv \
        --output-dir _runs/classification_fold${fold} \
        --fold ${fold} \
        --base resnet50 \
        --resume _runs/classification_fold${fold}/model.best.p \
        --print-model 0 \
        --n-tta 4 \
        --submission
done
