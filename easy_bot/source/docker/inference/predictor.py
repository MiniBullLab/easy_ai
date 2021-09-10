import sys
import os
import argparse
import logging
import warnings
import io
import json
import subprocess
import time
import ast

import cv2
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv import data, utils
import gluonnlp as nlp
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore",category=FutureWarning)

# sys.path.append(os.path.join(os.path.dirname(__file__), '/opt/ml/code/package'))

import pickle
# from io import StringIO
from timeit import default_timer as timer
from collections import Counter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask

# import sagemaker
# from sagemaker import get_execution_role, local, Model, utils, fw_utils, s3
# import boto3
import tarfile

# import pandas as pd

prefix = '/opt/ml/'
model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

hyperparameters_str = os.environ.get('HYPERPARAMETERS', '{}')
hyperparameters = ast.literal_eval(hyperparameters_str) # convert string expr to dict
image_size = int(hyperparameters.get("IMAGE_SIZE", "256"))
image_size_crop = int(round(image_size * 0.875))
print(f"image_size={image_size}, image_size_crop={image_size_crop}")
to_gray = int(hyperparameters.get("TO_GRAY", "0"))
print(f"to_gray={to_gray}")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    text_vocab = None
    tag_vocab = None
    tokenizer = None
    task_type = os.environ.get("TASK_TYPE")

    @classmethod
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.model == None:
            print(os.listdir("/opt/ml/model/"))
            if self.task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
                self.model = gluon.nn.SymbolBlock.imports(
                    "/opt/ml/model/custom_model-symbol.json", ['data'],
                    "/opt/ml/model/custom_model-0000.params", ctx=ctx)
            elif self.task_type == "NAMED_ENTITY_RECOGNITION":
                self.model = gluon.nn.SymbolBlock.imports(
                    "/opt/ml/model/custom_model-symbol.json", ['data0', 'data1', 'data2'],
                    "/opt/ml/model/custom_model-0000.params", ctx=ctx)
                with open('/opt/ml/model/text_vocab.json', 'r') as fp:
                    text_vocab_str = fp.read()
                    self.text_vocab = nlp.Vocab.from_json(text_vocab_str)
                with open('/opt/ml/model/tag_vocab.json', 'r') as fp:
                    tag_vocab_str = fp.read()
                    self.tag_vocab = nlp.Vocab.from_json(tag_vocab_str)
                self.tokenizer = nlp.data.BERTTokenizer(vocab=self.text_vocab, lower=True)
            self.model.hybridize(static_alloc=True, static_shape=True)
        return self.model

    @classmethod
    def predict(self, input_np):
        if self.task_type == "IMAGE_CLASSIFICATION":
            return self.image_classification_predict(input_np)
        elif self.task_type == "OBJECT_DETECTION":
            return self.object_detection_predict(input_np)
        elif self.task_type == "NAMED_ENTITY_RECOGNITION":
            return self.named_entity_recognition_predict(input_np)
        else:
            raise RuntimeError("Unknown task type {}".format(self.task_type))

    @classmethod
    def image_classification_predict(self, input_np):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        tic = time.time()

        if to_gray:
            input_np = cv2.cvtColor(input_np, cv2.COLOR_BGR2GRAY)
            input_np = cv2.cvtColor(input_np, cv2.COLOR_GRAY2BGR)

        input_nd = mx.nd.array(input_np, mx.cpu())
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size_crop),
        ])
        input_np = transform(input_nd).asnumpy()
        
        input_np = input_np.astype(np.float32)
        input_np = input_np - [123.675, 116.28, 103.53]
        input_np /= [58.395, 57.12, 57.375]
        input_np = input_np.transpose(2, 0, 1)
        input_np = np.expand_dims(input_np, axis=0)

        toc = time.time()
        print(f"0 - elapsed: {(toc-tic)*1000.0} ms")
        
        net = self.get_model()

        toc = time.time()
        print(f"0.1 - elapsed: {(toc-tic)*1000.0} ms")

        input_nd = mx.nd.array(input_np, ctx)
        # input_nd = input_np.as_in_context(ctx)

        toc = time.time()
        print(f"0.2 - elapsed: {(toc-tic)*1000.0} ms")

        res = net(input_nd)

        toc = time.time()
        print(f"0.3 - elapsed: {(toc-tic)*1000.0} ms")

        prob = mx.nd.softmax(res)

        toc = time.time()
        print(f"0.4 - elapsed: {(toc-tic)*1000.0} ms")

        prob = prob.asnumpy()

        toc = time.time()
        print(f"1 - elapsed: {(toc-tic)*1000.0} ms")

        prob_sqr = (prob * prob) / np.sum(prob * prob)
        prob = prob_sqr
        prob_sqr = (prob * prob) / np.sum(prob * prob)
        prob = prob_sqr

        toc = time.time()
        print(f"2 - elapsed: {(toc-tic)*1000.0} ms")

        print("prob={}".format(str(prob)))
        clsidx = np.argmax(prob)
        classes = []
        with open("/opt/ml/model/classes.txt", "r") as fp:
            classes = fp.readlines()
            classes = [l.strip() for l in classes]
        clsidx = int(clsidx)
        prob = prob.flatten().tolist()
        print("classes={}".format(str(classes)))
        print("prob={}".format(str(prob)))
        results = [{"Class": cls, "Probability": p} for cls, p in zip(classes, prob)]
        print("results={}".format(str(results)))
        output_dict = {"ClassIndex": clsidx, "Results": results}
        output_dict_str = json.dumps(output_dict)
        print(output_dict_str)

        toc = time.time()
        print(f"3 - elapsed: {(toc-tic)*1000.0} ms")

        # Convert from numpy back to CSV
        out = io.StringIO()
        # pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
        out.write(output_dict_str)
        result = out.getvalue()
        return result

    @classmethod
    def object_detection_predict(self, input_np):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        import boto3

        net = self.get_model()
        # Operator _cvimresize is not implemented for GPU
        x = mx.nd.array(input_np, mx.cpu())
        origin_shape = x.shape
        x, img = data.transforms.presets.yolo.transform_test(x, short=416)
        x = x.as_in_context(ctx)
        class_IDs, scores, bounding_boxs = net(x)

        class_IDs = class_IDs.asnumpy()
        scores = scores.asnumpy()
        bounding_boxs = bounding_boxs.asnumpy()

        print(f"shape of x: {x.shape} {x.dtype}")
        print(f"origin_shape: {origin_shape}")
        print(f"shape of class_IDs: {class_IDs.shape} {class_IDs.dtype}")
        print(f"shape of scores: {scores.shape} {scores.dtype}")
        print(f"shape of bounding_boxs: {bounding_boxs.shape} {bounding_boxs.dtype}")
        
        with open("/opt/ml/model/classes.txt", "r") as f:
            classes = f.read().splitlines()

        result_dict = {"Result":
                           {"class_IDs": [int(i) for i in class_IDs.squeeze().tolist() if i != -1.],
                            "classes": classes,
                            "threshold": 0.5,
                            "scores": [i for i in scores.squeeze().tolist() if i != -1.],
                            "bounding_boxs": [(i * (origin_shape[0] / 416)).tolist() for i in bounding_boxs.squeeze() if list(i) != [-1.] * 4]},
                            "height": origin_shape[0],
                            "width": origin_shape[1]
                       }

        return json.dumps(result_dict)

    @classmethod
    def named_entity_recognition_predict(self, input_np):
        """For the input, do the predictions and return them.
        """
        seq_len = 128
        net = self.get_model()

        print(input_np)


        def encode_as_input(sentence, seq_len, text_vocab, tag_vocab, null_tag_index=0):
            """Enocde a single sentence into numpy arrays as input to the BERTTagger model.

            Parameters
            ----------
            sentence: List[TaggedToken]
                A sentence as a list of tagged tokens.

            Returns
            -------
            np.array: token text ids (batch_size, seq_len)
            np.array: token types (batch_size, seq_len),
                    which is all zero because we have only one sentence for tagging.
            np.array: valid_length (batch_size,) the number of tokens until [SEP] token
            np.array: tag_ids (batch_size, seq_len)
            np.array: flag_nonnull_tag (batch_size, seq_len),
                    which is simply tag_ids != null_tag_index

            """
            NULL_TAG = 'X'
            # check whether the given sequence can be fit into `seq_len`.
            # print(''.join([t.text[0] for t in sentence]))
            assert len(sentence) <= seq_len - 2, \
                'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
                .format(len(sentence), seq_len, ''.join([t[0] for t in sentence]))

            text_tokens = ([text_vocab.cls_token] + [token for token in sentence] +
                           [text_vocab.sep_token])
            padded_text_ids = (text_vocab.to_indices(text_tokens)
                               + ([text_vocab[text_vocab.padding_token]]
                                  * (seq_len - len(text_tokens))))

            tags = [NULL_TAG] + ['X' for token in sentence] + [NULL_TAG]
            padded_tag_ids = (tag_vocab.to_indices(tags)
                              + [tag_vocab[NULL_TAG]] * (seq_len - len(tags)))

            assert len(text_tokens) == len(tags)
            assert len(padded_text_ids) == len(padded_tag_ids)
            assert len(padded_text_ids) == seq_len

            valid_length = len(text_tokens)

            # in sequence tagging problems, only one sentence is given
            token_types = [0] * seq_len

            np_tag_ids = mx.nd.array(padded_tag_ids, dtype='int32', ctx=ctx)
            # gluon batchify cannot batchify numpy.bool? :(
            flag_nonnull_tag = (np_tag_ids != null_tag_index).astype('int32')

            return (mx.nd.array(padded_text_ids, dtype='int32', ctx=ctx),
                    mx.nd.array(token_types, dtype='int32', ctx=ctx),
                    # np.array(valid_length, dtype='int32'),
                    mx.nd.array([valid_length], dtype='int32', ctx=ctx),
                    np_tag_ids,
                    flag_nonnull_tag)
        
        sentence = str(input_np)
        sentence = self.tokenizer(sentence)
        data = encode_as_input(sentence, seq_len, self.text_vocab, self.tag_vocab)

        text_ids, token_types, valid_length, tag_ids, _ = data # [x.astype('float32').as_in_context(ctx) for x in data]
        text_ids = text_ids.expand_dims(axis=0).astype('float32').as_in_context(ctx)
        token_types = token_types.expand_dims(axis=0).astype('float32').as_in_context(ctx)
        out = net(text_ids, token_types, valid_length.astype('float32'))

        # convert results to numpy arrays for easier access
        np_text_ids = text_ids.astype('int32').asnumpy().flatten()
        np_pred_tags = out.argmax(axis=-1).asnumpy().flatten()
        np_valid_length = valid_length.astype('int32').asnumpy().flatten()
        np_true_tags = tag_ids.asnumpy().flatten()

        text = [self.text_vocab.idx_to_token[int(t)] for t in np_text_ids]
        tags = [self.tag_vocab.idx_to_token[int(t)] for t in np_pred_tags]

        print(text)
        print(tags)

        def token2tag(t):
            if t in ["O", "X"]:
                return t
            else:
                return t[1:].strip("-")
        # labels = list(set([token2tag(token) for token in self.tag_vocab.idx_to_token]))
        # labels = filter(lambda x: len(x)>1, labels) # filter zero length labels
        # labels = [_ for _ in labels]
        labels = []
        result = ""
        last_t, last_last_t = None, None
        for c, t in zip(text, tags):
            if c in ["[CLS]", "[UNK]"]:
                c = ""
                continue
            elif c in ["[SEP]", "[PAD]"]:
                c = ""
                if last_t not in ["O", "X"]:
                    result += "}}"
                print(f"BREAKING: '{last_last_t}' '{last_t}' '{t}' '{c}'")
                print(result)
                break
            t = token2tag(t)
            print(c, t, last_t)
            if c.isalpha() or c.startswith("##"):
                c = " " + c
            labels.append(t)
            if last_t is None:
                if t in ["O", "X"]:
                    result += c
                else:
                    result += "{{" + t + ":" + c
            else:
                if last_t == t:
                    result += c
                elif t in ["O", "X"]:
                    result += "}}" + c
                else:
                    if last_t in ["O", "X"]:
                        result += "{{" + t + ":" + c
                    else:
                        result += "}}" + "{{" + t + ":" + c
            last_last_t = last_t
            last_t = t
        labels = list(set(labels))
        labels = filter(lambda x: len(x)>1, labels) # filter zero length labels
        labels = [_ for _ in labels]
        results = [result,]
        
        results = {
            "Labels": labels,
            "Results": results, 
        }
        return json.dumps(results)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    tic = time.time()
    
    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        data_np = np.asarray(data['data'])
    elif flask.request.content_type == 'image/jpeg':
        data = flask.request.data
        print("len(data)={}".format(len(data)))
        data_np = np.fromstring(data, dtype=np.uint8)
        print("data_np.shape={}".format(str(data_np.shape)))
        print(' '.join(['{:x}'.format(d) for d in data_np[:20].tolist()]), flush=True)
        data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
        data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)
    else:
        return flask.Response(response='This predictor only supports JSON data and JPEG image data',
                              status=415, mimetype='text/plain')
    print(data_np.shape)

    toc = time.time()
    print(f"0 - invocations: {(toc - tic) * 1000.0} ms")

    # Do the prediction
    response = ScoringService.predict(data_np)
    ret = flask.Response(response=response, status=200, mimetype='application/json')

    toc = time.time()
    print(f"1 - invocations: {(toc - tic) * 1000.0} ms")

    return ret
