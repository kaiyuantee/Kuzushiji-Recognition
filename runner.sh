#!/bin/bash

for fold in {0..4};
do
	echo "this is fold ${fold}"
	#train the frcnn model
	python -m kuzushiji.object_detection.main --output-dir _runs/detection_fold${fold} \
	--fold ${fold} --model fasterrcnn_resnet50_fpn
	#oof predictions
	python -m kuzushiji.object_detection.main --output-dir _runs/detection_fold${fold} \
        --fold ${fold} --model fasterrcnn_resnet50_fpn --resume _runs/detection_fold${fold}/model_best.p \
        --test-only
	#inference
	python -m kuzushiji.object_detection.main --output-dir _runs/detection_fold${fold} \
        --fold ${fold} --model fasterrcnn_resnet50_fpn --resume _runs/detection_fold${fold}/model_best.p \
        --submission
done

#blend the predictions from all 5 folds
python -c "
import pandas as pd;
pd.concat([pd.read_csv(f'_runs/segment-fold{i}/clf_gt.csv') for i in range(5)]).to_csv('_runs/detection_gt.csv')
"
