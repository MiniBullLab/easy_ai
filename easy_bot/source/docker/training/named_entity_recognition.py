#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""4. Transfer Learning with Your Own Text Dataset
=======================================================

Transfer learning is a technique that addresses this problem.
The idea is simple: we can start training with a pre-trained model,
instead of starting from scratch.
As Isaac Newton said, "If I have seen further it is by standing on the
shoulders of Giants".
"""

def train():
    import argparse
    import os
    import boto3
    import mxnet as mx
    import random
    import shutil
    import ast

    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])

    test_size = 0.3

    data_bucket = os.environ.get("DATA_BUCKET")
    data_prefix = os.environ.get("DATA_PREFIX")
    model_bucket = os.environ.get("MODEL_BUCKET")
    model_prefix = os.environ.get("MODEL_PREFIX")

    print("DATA_BUCKET={}".format(data_bucket))
    print("DATA_PREFIX={}".format(data_prefix))
    print("MODEL_BUCKET={}".format(model_bucket))
    print("MODEL_PREFIX={}".format(model_prefix))

    basename = ''

    code_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(code_dir, 'data')
    output_dir = '/opt/ml/output'
    model_dir = '/opt/ml/model'

    for dirname in [train_dir, output_dir, model_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Download all objects
    key = "{}/train.txt".format(data_prefix)
    train_fullpath = "{}/train.txt".format(train_dir)
    print("Downloading from s3://{}/{} to {}".format(data_bucket, key, train_fullpath))
    s3.download_file(data_bucket, key, train_fullpath)

    # Generate validation set
    import shutil
    val_fullpath = "{}/val.txt".format(train_dir)
    shutil.copyfile(train_fullpath, val_fullpath)
    print(f"Copied from {train_fullpath} to {val_fullpath}")

    with open(train_fullpath, "rt") as fp:
        line = fp.readline()
        if "{{" in line:
            tmp_fullpath = "{}/tmp.txt".format(train_dir)
            shutil.copyfile(train_fullpath, tmp_fullpath)
            from ner_utils import bosonnlp_to_bio2
            print(f"Converting to BIO2 format: {train_fullpath} and {val_fullpath}.", flush=True)
            bosonnlp_to_bio2(tmp_fullpath, train_fullpath, val_fullpath)
            print(f"Done", flush=True)

    ################################################################################
    # Hyperparameters
    # ----------
    #
    # First, let's import all other necessary libraries.

    import mxnet as mx
    import numpy as np
    import os, time, shutil
    import mxnet as mx
    import gluonnlp as nlp
    import logging

    from mxnet import gluon, image, init, nd
    from mxnet import autograd as ag
    from mxnet.gluon import nn

    from ner_utils import BERTTaggingDataset, BERTTagger, attach_prediction, convert_arrays_to_text

    ################################################################################
    # We set the hyperparameters as following:

    hyperparameters = ast.literal_eval(os.environ["HYPERPARAMETERS"]) # convert string expr to dict
    num_epochs = int(hyperparameters.get("EPOCHS", "80"))
    language = hyperparameters.get("LANGUAGE", "zh")
    if language == "zh":
        bert_dataset_name = "wiki_cn_cased"
    elif language == "en":
        bert_dataset_name = "book_corpus_wiki_en_uncased"
    else:
        raise RuntimeError(f"Unexpected LANGUAGE={language}.")
    print(f"LANGUAGE={language}, bert_dataset_name={bert_dataset_name}")

    batch_size = 64
    dropout_prob = 0.1
    warmup_ratio = 1.0 / float(num_epochs)
    learning_rate = 0.01
    momentum = 0.9
    wd = 0.0001
    optimizer = 'sgd'
    save_checkpoint_prefix = 'ner'
    seq_len = 128

    # provide random seed for every RNGs we use
    seed = 0xdeadbeef
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

    ctx = mx.gpu(0)

    def get_bert_model(bert_model, bert_dataset_name, ctx, dropout_prob):
        """Get pre-trained BERT model."""
        bert_model, text_vocab = nlp.model.get_model(
            name=bert_model,
            dataset_name=bert_dataset_name,
            pretrained=True,
            ctx=ctx,
            use_pooler=False,
            use_decoder=False,
            use_classifier=False,
            dropout=dropout_prob,
            embed_dropout=dropout_prob)
        return bert_model, text_vocab


    print('Loading BERT model...')
    bert_model, text_vocab = get_bert_model("bert_12_768_12", bert_dataset_name, ctx, 0.1)
    if not os.path.exists("models"):
        os.makedirs("models")
    with open('models/text_vocab.json', 'w') as fp:
        fp.write(text_vocab.to_json())

    dataset = BERTTaggingDataset(text_vocab, 
                                 "./data/train.txt", 
                                 "./data/val.txt",
                                 seq_len, True)

    with open('models/tag_vocab.json', 'w') as fp:
        fp.write(dataset.tag_vocab.to_json())

    train_data_loader = dataset.get_train_data_loader(batch_size=batch_size)
    dev_data_loader = dataset.get_dev_data_loader(batch_size=batch_size)

    net = BERTTagger(bert_model, dataset.num_tag_types, dropout_prob)
    net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize(static_alloc=True)

    step_size = batch_size
    num_train_steps = int(len(dataset.train_inputs) / step_size * num_epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    optimizer_params = {'learning_rate': learning_rate, 'momentum': momentum, 'wd': wd}
    trainer = mx.gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    # collect differentiable parameters
    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    def train(data_loader, start_step_num, epoch_index, num_epochs):
        """Training loop."""
        step_num = start_step_num
        print(f'current starting step num: {step_num}')
        for batch_id, (text_ids, _, _, tag_ids, flag_nonnull_tag, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=True)):
            print('training on epoch {}/{}, batch: {}/{}'.format(epoch_index, num_epochs, batch_id, len(data_loader)), end=', ')

            # step size adjustments
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = learning_rate * step_num / num_warmup_steps
            else:
                offset = ((step_num - num_warmup_steps) * learning_rate /
                          (num_train_steps - num_warmup_steps))
                new_lr = learning_rate - offset
            trainer.set_learning_rate(new_lr)
            print('lr: {:6f}'.format(new_lr), end=', ')

            with mx.autograd.record():
                loss_value = loss_function(out, tag_ids,
                                           flag_nonnull_tag.expand_dims(axis=2)).mean()

            loss_value.backward()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.step(1)

            pred_tags = out.argmax(axis=-1)
            print('loss: {:3f}'.format(loss_value.asscalar()), end=', ')

            num_tag_preds = flag_nonnull_tag.sum().asscalar()
            print('accuracy: {:3f}'.format(((pred_tags == tag_ids) * flag_nonnull_tag).sum().asscalar() / num_tag_preds))

            # print("text_ids:")
            # print(text_ids[0].asnumpy().astype(np.int32))
            # print("tag_ids:")
            # print(tag_ids[0].asnumpy().astype(np.int32))
            # print("out:")
            # print(np.argmax(out[0].asnumpy(), axis=1).astype(np.int32))

        return step_num

    def evaluate(data_loader, verbose=False):
        """Eval loop."""
        predictions = []

        for batch_id, (text_ids, _, valid_length, tag_ids, _, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=False)):
            print(f'evaluating on batch index: {batch_id}/{len(data_loader)}')

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            np_true_tags = tag_ids.asnumpy()

            predictions += convert_arrays_to_text(text_vocab, dataset.tag_vocab, np_text_ids,
                                                  np_true_tags, np_pred_tags, np_valid_length)

        all_true_tags = [[entry.true_tag for entry in entries] for entries in predictions]
        all_pred_tags = [[entry.pred_tag for entry in entries] for entries in predictions]
        seqeval_f1 = 0.0 # seqeval.metrics.f1_score(all_true_tags, all_pred_tags)
        return seqeval_f1

    best_dev_f1 = 0.0
    last_test_f1 = 0.0
    best_epoch = -1
    last_epoch_step_num = 0
    model_prefix = os.path.join(model_dir, "custom_model")

    for epoch_index in range(num_epochs):
        last_epoch_step_num = train(train_data_loader, last_epoch_step_num, epoch_index, num_epochs)
        train_f1 = 0.0 # evaluate(train_data_loader, verbose=True)
        # print('epoch {:03d}, train f1: {:3f}'.format(epoch_index, train_f1))
        # dev_f1 = evaluate(dev_data_loader, verbose=False)
        # print('epoch {:03d}, val f1: {:3f}, previous best val f1: {:3f}'.format(epoch_index, dev_f1, best_dev_f1))

    # save params
    print(f'saving current checkpoint to: {model_prefix}-0000.params')
    net.export(model_prefix, 0)


    # Create archive file for uploading to s3 bucket
    model_path = os.path.join(model_dir, "model.tar.gz")
    print("Creating compressed tar archive {}".format(model_path))
    import tarfile
    tar = tarfile.open(model_path, "w:gz")
    for name in ["{}-0000.params".format(model_prefix),
                 "{}-symbol.json".format(model_prefix),
                 "models/tag_vocab.json",
                 "models/text_vocab.json"]:
        basename = os.path.basename(name)
        tar.add(name, arcname=basename)
    tar.close()

    # Upload to S3 bucket
    print("Uploading tar archive {} to s3://{}/{}/".format(model_path, model_bucket, data_prefix))
    s3.upload_file(model_path, model_bucket, "{}/model.tar.gz".format(data_prefix))
