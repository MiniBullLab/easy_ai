#!/usr/bin/env python

def train():

    import argparse
    import boto3
    import mxnet as mx
    import gluoncv as gcv
    import numpy as np
    import os, time, random, shutil
    import logging
    import glob
    import ast

    import asyncio
    import functools
    import time

    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET

    from mxnet import gluon, image, init, nd
    from mxnet import autograd
    from mxnet.gluon.data import DataLoader
    from mxnet.gluon.data.vision import transforms
    from gluoncv import model_zoo
    from gluoncv.data import VOCDetection
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms import presets
    from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
    from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
    from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
    from gluoncv.utils import LRScheduler, LRSequential

    from voc_utils import convert_trainval
    # from s3_utils import list_objects, download_objects

    test_size = 0.3

    data_bucket = os.environ.get("DATA_BUCKET")
    data_prefix = os.environ.get("DATA_PREFIX")
    model_bucket = os.environ.get("MODEL_BUCKET")
    # model_prefix = os.environ.get("MODEL_PREFIX")

    print("DATA_BUCKET={}".format(data_bucket))
    print("DATA_PREFIX={}".format(data_prefix))
    print("MODEL_BUCKET={}".format(model_bucket))
    # print("MODEL_PREFIX={}".format(model_prefix))

    hyperparameters = ast.literal_eval(os.environ["HYPERPARAMETERS"]) # convert string expr to dict
    epochs = int(hyperparameters.get("EPOCHS", "80"))

    data_dir = '/opt/ml/input/data'
    output_dir = '/opt/ml/output'
    model_dir = '/opt/ml/model'

    for dirname in [data_dir, output_dir, model_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Download all objects
    print("Collecting objects in s3://{}/{}".format(data_bucket, data_prefix))
    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
    all_keys = []
    task_list = []
    response = s3.list_objects_v2(Bucket=data_bucket, Prefix=data_prefix)
    while True:
        keys = response["Contents"]
        for ctx in keys:
            key = ctx["Key"]
            if key.endswith("/"):
                print(f"Skipping {data_bucket}/{key}.")
                continue
            relpath = key[len(data_prefix):].strip('/')
            dirname = os.path.dirname(relpath)
            basename = os.path.basename(relpath)
            fulldirname = os.path.join(data_dir, "VOC2018", dirname)
            fullpath = os.path.join(fulldirname, basename)
            if not os.path.exists(fulldirname):
                os.makedirs(fulldirname)
            # s3.download_file(data_bucket, key, fullpath)
            task_list.append((data_bucket, key, fullpath))
            all_keys.append(fullpath)
        truncated = response["IsTruncated"]
        if not truncated:
            break
        token = response["NextContinuationToken"]
        response = s3.list_objects_v2(Bucket=data_bucket, Prefix=data_prefix, ContinuationToken=token)

    print("Downloading {} objects from s3://{}/{}/ to {}".format(len(all_keys), data_bucket, data_prefix, data_dir))
    
    # async def main_task():
    #     loop = asyncio.get_running_loop()
    #     objects = await asyncio.gather(
    #         *[
    #             loop.run_in_executor(None, functools.partial(s3.download_file, data_bucket, key, fullpath))
    #             for data_bucket, key, fullpath in task_list
    #         ]
    #     )
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main_task())

    tic = time.time()
    for data_bucket, key, fullpath in task_list:
        toc = time.time()
        if (toc - tic) > 10:
            print(f"Downloading s3://{data_bucket}/{key} to {fullpath}")
            tic = time.time()
        s3.download_file(data_bucket, key, fullpath)

    print("Downloaded {} objects.".format(len(all_keys)))
    print(os.listdir(os.path.join(data_dir, "VOC2018")))

    # Generate validation set
    trainval_file = os.path.join(data_dir, 'VOC2018', 'ImageSets', 'Main', 'trainval.txt')
    if os.path.exists(trainval_file):
        with open(trainval_file, 'r') as f:
            prefixs = f.readlines()
        random.shuffle(prefixs)
        train_file = os.path.join(data_dir, 'VOC2018', 'ImageSets', 'Main', 'train.txt')
        with open(train_file, 'w') as f:
            f.writelines(prefixs[int(len(prefixs) * test_size):])
        val_file = os.path.join(data_dir, 'VOC2018', 'ImageSets', 'Main', 'val.txt')
        with open(val_file, 'w') as f:
            f.writelines(prefixs[:int(len(prefixs) * test_size)])
    else:
        # search for xxx_train.txt and xxx_val.txt files
        voc_data_dir = os.path.join(data_dir, 'VOC2018')
        convert_trainval(voc_data_dir)

    classes_path = os.path.join(data_dir, 'VOC2018', 'classes.txt')
    classes = []
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            classes = f.read().splitlines()
        classes = sorted(list(classes))
    else:
        classes = set()
        all_xml = glob.glob(os.path.join(data_dir, 'VOC2018', 'Annotations', '*.xml'))
        for each_xml_file in all_xml:
            tree = ET.parse(each_xml_file)
            root = tree.getroot()
            for child in root:
                if child.tag == 'object':
                    for item in child:
                        if item.tag == 'name':
                            classes.add(item.text)
        classes = sorted(list(classes))
        with open(classes_path, 'w') as f:
            f.writelines([line+'\n' for line in classes])
    print(f"classes={classes}")

    ###########################################################
    # training job
    ###########################################################
    def parse_args():
        parser = argparse.ArgumentParser(description='Train YOLO network.')
        parser.add_argument('--data-shape', type=int, default=416,
                            help="Input data shape for evaluation, use 320, 416, 608... " +
                                 "Training is with random shapes from (320 to 608).")
        parser.add_argument('--batch-size', type=int, default=16,
                            help='Training mini-batch size')
        parser.add_argument('--epochs', type=int, default=epochs,
                            help='Training epochs.')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='Learning rate, default is 0.0001')
        parser.add_argument('--lr-mode', type=str, default='step',
                            help='learning rate scheduler mode. options are step, poly and cosine.')
        parser.add_argument('--lr-decay', type=float, default=0.1,
                            help='decay rate of learning rate. default is 0.1.')
        parser.add_argument('--lr-decay-period', type=int, default=0,
                            help='interval for periodic learning rate decays. default is 0 to disable.')
        parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                            help='epochs at which learning rate decays. default is 160,180.')
        parser.add_argument('--warmup-lr', type=float, default=0.0,
                            help='starting warmup learning rate. default is 0.0.')
        parser.add_argument('--warmup-epochs', type=int, default=0,
                            help='number of warmup epochs.')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='SGD momentum, default is 0.9')
        parser.add_argument('--wd', type=float, default=0.0005,
                            help='Weight decay, default is 5e-4')
        parser.add_argument('--log-interval', type=int, default=100,
                            help='Logging mini-batch interval. Default is 100.')
        parser.add_argument('--save-prefix', type=str, default='',
                            help='Saving parameter prefix')
        parser.add_argument('--val-interval', type=int, default=1,
                            help='Epoch interval for validation, increase the number will reduce the '
                                 'training time if validation is slow.')
        parser.add_argument('--num-samples', type=int, default=-1,
                            help='Training images. Use -1 to automatically get the number.')
        parser.add_argument('--no-wd', action='store_true',
                            help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
        parser.add_argument('--mixup', action='store_true',
                            help='whether to enable mixup.')
        parser.add_argument('--no-mixup-epochs', type=int, default=20,
                            help='Disable mixup training if enabled in the last N epochs.')
        parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')

        args = parser.parse_args(args=[])
        return args


    def get_dataset(args):
        """Get dataset."""
        class CustomVOC(VOCDetection):
            CLASSES = classes
            def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
                super(CustomVOC, self).__init__(root, splits, transform, index_map, preload_label)

        train_dataset = CustomVOC(root=data_dir, splits=[(2018, 'train')])
        val_dataset = CustomVOC(root=data_dir, splits=[(2018, 'val')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        if args.num_samples < 0:
            args.num_samples = len(train_dataset)
        if args.mixup:
            from gluoncv.data import MixupDetection
            train_dataset = MixupDetection(train_dataset)
        return train_dataset, val_dataset, val_metric


    def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
        """Get dataloader."""
        width, height = data_shape, data_shape
        # stack image, all targets generated
        batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers,
            timeout=600)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers,
            timeout=600)
        return train_loader, val_loader


    def save_params(net, best_map, current_map, epoch, prefix):
        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            net.export(prefix, 0)


    def validate(net, val_data, ctx, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        mx.nd.waitall()
        net.hybridize()
        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return eval_metric.get()


    def train(net, train_data, val_data, eval_metric, batch_size, ctx, args):
        """Training pipeline"""
        net.collect_params().reset_ctx(ctx)
        if args.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if args.label_smooth:
            net._target_generator._label_smooth = True

        if args.lr_decay_period > 0:
            lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
        lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
        num_batches = args.num_samples // args.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=args.lr,
                        nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(args.lr_mode, base_lr=args.lr,
                        nepochs=args.epochs - args.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=args.lr_decay, power=2),
        ])
        trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
            kvstore='local')

        # targets
        sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        l1_loss = gluon.loss.L1Loss()

        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')

        # set up logger
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = args.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        logger.info(args)
        logger.info('Start training from [Epoch 0]')
        best_map = [0]
        for epoch in range(0, args.epochs):
            if args.mixup:
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= args.epochs - args.no_mixup_epochs:
                    try:
                        train_data._dataset.set_mixup(None)
                    except AttributeError:
                        train_data._dataset._data.set_mixup(None)

            tic = time.time()
            btic = time.time()
            mx.nd.waitall()
            net.hybridize()
            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    autograd.backward(sum_losses)
                trainer.step(batch_size)
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name1, loss1 = obj_metrics.get()
                    name2, loss2 = center_metrics.get()
                    name3, loss3 = scale_metrics.get()
                    name4, loss4 = cls_metrics.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, trainer.learning_rate, args.batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

            name1, loss1 = obj_metrics.get()
            name2, loss2 = center_metrics.get()
            name3, loss3 = scale_metrics.get()
            name4, loss4 = cls_metrics.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, best_map, current_map, epoch, args.save_prefix)


    args = parse_args()
    model_prefix = os.path.join(model_dir, "custom_model")
    args.save_prefix = model_prefix

    num_gpus = 1
    num_workers = 2
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    net = model_zoo.yolo3_darknet53_coco(pretrained=True)
    # net = model_zoo.yolo3_mobilenet1_0_coco(pretrained=True)
    net.reset_class(classes)
    net.initialize()

    batch_size = args.batch_size
    train_dataset, val_dataset, eval_metric = get_dataset(args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, batch_size, num_workers, args)

    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
    ###########################################################

    # Create archive file for uploading to s3 bucket
    model_path = os.path.join(model_dir, "model.tar.gz")
    print("Creating compressed tar archive {}".format(model_path))
    import tarfile
    tar = tarfile.open(model_path, "w:gz")
    for name in ["{}-0000.params".format(model_prefix),
                 "{}-symbol.json".format(model_prefix),
                 os.path.join(data_dir, 'VOC2018', "classes.txt")]:
        basename = os.path.basename(name)
        tar.add(name, arcname=basename)
    tar.close()

    # Upload to S3 bucket
    model_prefix = os.environ.get("MODEL_PREFIX")
    print("Uploading tar archive {} to s3://{}/{}/".format(model_path, model_bucket, model_prefix))
    s3.upload_file(model_path, model_bucket, "{}/model.tar.gz".format(model_prefix))
