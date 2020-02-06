#!/bin/bash

# Create the pseduolabels first
python -m kuzushiji.object_classification.pseudo \
    _runs/classification-fold[0-4]/test_detailed.csv.gz \
    _runs/pseudolabels.csv.gz

#Train model with pseudolabels (resume from best weight)
for fold in {0..4};
do
    echo "Training with pseudolabels now"
    echo "This is Fold ${fold}"

    python -m kuzushiji.object_classification.main \
        _runs/detection_gt.csv \
        --output-dir _runs/classification-fold${fold}-pseudo --fold ${fold} --print-model 0 \
        --resume _runs/classification-fold${fold}/model_best.p \
        --max-targets 256 --benchmark 1 --frozen-start 1 \
        --base resnet50 --workers 4 --batch-size 12 --lr 1.6e-3 --opt-level O1 \
        --pseudolabels _runs/pseudolabels.csv.gz \
        --repeat-train 3 \
        --epochs 5
    python -m kuzushiji.object_classification.main \
        _runs/detection_gt.csv \
        --output-dir _runs/classification-fold${fold}-pseudo --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnet50 --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/classification-fold${fold}-pseudo/model_best.pth \
        --n-tta 4 --test-only
    python -m kuzushiji.object_classification.main \
        _runs/detection_fold0/test_predictions.csv \
        --output-dir _runs/classification-fold${fold}-pseudo --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnet50 --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/classification-fold${fold}-pseudo/model_best.pth \
        --n-tta 4 --submission
done
