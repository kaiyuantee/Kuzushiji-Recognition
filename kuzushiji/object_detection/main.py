import pandas as pd
import torch.utils.data
import torch.nn as nn
import argparse
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import frcnn
from frcnn.rpn import AnchorGenerator
from frcnn.transform import GeneralizedRCNNTransform
from frcnn.faster_rcnn import FastRCNNPredictor
from .engine import train_one_epoch, evaluate
from .utils import *
from .augmentation import Dataset, augmentation
from ..data_utils import DATA_ROOT, TRAIN_ROOT, TEST_ROOT, load_train_valid_df


class Transformation(GeneralizedRCNNTransform):

    def __init__(self, image_mean, image_std):
        nn.Module.__init__(self)
        self.image_mean = image_mean
        self.image_std = image_std

    def resize(self, image, target):
        return image, target


def build_model(name: str, pretrained: bool, nms_threshold: float):
    anchor_sizes = [12, 24, 32, 64, 96]
    model = frcnn.__dict__[name](pretrained=pretrained,
                                 rpn_anchor_generator=AnchorGenerator(sizes=tuple((s,) for s in anchor_sizes),
                                                                      aspect_ratios=tuple(
                                                                          (0.5, 1.0, 2.0) for _ in anchor_sizes)
                                                                      ),
                                 box_detections_per_img=1000,
                                 box_nms_thresh=nms_threshold)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=2)
    model.transform = Transformation(image_mean=model.transform.image_mean,
                                     image_std=model.transform.image_std)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='fasterrcnn_resnet50_fpn', help='model')  # resnet50/152?
    arg('--device', default='cuda', help='device')  # cuda for gpu
    arg('--batch-size', default=16, type=int)  # batchsize
    arg('--workers', default=4, type=int,
        help='number of data loading workers')  # workers
    arg('--lr', default=0.01, type=float, help='initial learning rate')  # learing rate
    arg('--momentum', default=0.9, type=float, help='momentum')  # optimizer momentum
    arg('--wd', '--weight-decay', default=1e-4, type=float,
        help='weight decay (default: 1e-4)', dest='weight_decay')  # optimizer weight decay
    arg('--epochs', default=45, type=int,
        help='number of total epochs to run')  # epochs
    arg('--lr-steps', default=[35], nargs='+', type=int,
        help='decrease lr every step-size epochs')  # learning rate scheduler
    arg('--lr-gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')  # lr scheduler rate
    arg('--cosine', type=int, default=0,
        help='cosine lr schedule (disabled step lr schedule)')  # cosine lr scheduler
    arg('--print-freq', default=100, type=int, help='print frequency')  # print freq
    arg('--output-dir', help='path where to save')  # output directory after training
    arg('--resume', help='resume from checkpoint')  # resume training from checkpoint
    arg('--test-only', help='Only test the model', action='store_true')  # testing only without submission
    arg('--submission', help='Create test predictions', action='store_true')  # submission
    arg('--pretrained', type=int, default=0,
        help='Use pre-trained models from the modelzoo')  # pretrained models from modelzoo
    arg('--score-threshold', type=float, default=0.5)  # score threshold for detection
    arg('--nms-threshold', type=float, default=0.25)  # non max suppresion threshold for detection
    arg('--repeat-train-step', type=int, default=2)  # repeat train

    # fold parameters
    arg('--fold', type=int, default=0)  # how many folds
    arg('--n-folds', type=int, default=5)  # number of folds

    args = parser.parse_args()
    if args.test_only and args.submission:
        parser.error('Please select either test or submission')

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # Loading dataset
    print('...Loading Data Now...')

    df_train, df_valid = load_train_valid_df(args.fold, args.n_folds)  # from data_utils
    root = TRAIN_ROOT  # from data_utils
    if args.submission:
        df_valid = pd.read_csv(DATA_ROOT / 'sample_submission.csv')  # from data_utils
        df_valid['labels'] = ''
        root = TEST_ROOT
    dataset_train = Dataset(df_train, augmentation(train=True), root, skip_empty=False)
    dataset_test = Dataset(df_valid, augmentation(train=False), root, skip_empty=False)

    # Pytorch data loaders
    print('...Creating The Data Loaders Now...')
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch = torch.utils.data.BatchSampler(train_sampler,
                                                args.batch_size,
                                                drop_last=True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_sampler=train_batch,
                                                    num_workers=args.workers,
                                                    collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=args.workers,
                                                   collate_fn=collate_fn)

    # Create The Model
    print('...Creating Model Now...')
    model = build_model(args.model, args.pretrained, args.nms_threshold)
    model.to(device)

    params = [para for para in model.parameters() if para.requires_grad]  # requires grad?
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = None
    if args.cosine:
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_steps:
        lr_scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)
        print('Loaded from checkpoint {}'.format(args.resume))

    def save_eval_results(results):
        scores, clf_gt = results
        if output_dir:
            pd.DataFrame(scores).to_csv(output_dir / 'eval.csv', index=None)
            pd.DataFrame(clf_gt).to_csv(output_dir / 'clf_gt.csv', index=None)

    if args.test_only or args.submission:
        _, eval_results = evaluate(
            model, data_loader_test, device=device, output_dir=output_dir,
            threshold=args.score_threshold)
        if args.test_only:
            save_eval_results(eval_results)
        elif output_dir:
            pd.DataFrame(eval_results[1]).to_csv(
                output_dir / 'test_predictions.csv', index=None)
        return

    # Start Training
    print('...Training Session Begin...')
    best_f1 = 0
    start = time.time()
    for epoch in range(args.epochs):
        #         train_sampler.set_epoch(epoch)
        for _ in range(args.repeat_train_step):
            train_metrics = train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq)
        if lr_scheduler:
            lr_scheduler.step()
        if output_dir:
            # json_log_plots.write_event(output_dir, step=epoch, **train_metrics)
            save_on_master({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': (lr_scheduler.state_dict if lr_scheduler else None),
                            'args': args},
                           output_dir / 'checkpoint.p')
        # evaluation for every epoch
        eval_metrics, eval_results = evaluate(model, data_loader_test, device=device, output_dir=None,
                                              threshold=args.score_threshold)
        save_eval_results(eval_results)
        if output_dir:
            # json_log_plots.write_event(output_dir, step=epoch, **eval_metrics)
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                print('Updated best model with f1 of {}'.format(best_f1))
                save_on_master(
                    model.state_dict(),
                    output_dir / 'model_best.p')

    total_time = time.time() - start
    final = str(datetime.timedelta(seconds=int(total_time)))
    print('Trained for {} seconds'.format(final))


if __name__ == '__main__':
    main()
