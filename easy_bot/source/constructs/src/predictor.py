import io
import os
import sys
import time
import boto3
import tarfile
import shutil
import json
import ast
from dynamodb_controller import TaskTable
from model import task_type_map, task_type_map_s3, task_type_lut
from utils import get_datetime_str

taskTable = TaskTable()
s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
model_dict = {}

def invoke_endpoint_serverless(job_id, task_type, data_json):
    # Setting library paths.
    efs_path = "/mnt/ml"
    print(f"{os.listdir(efs_path)}")
    python_pkg_path = os.path.join(efs_path, "code/lib/python3.7/site-packages")
    sys.path.append(python_pkg_path)

    print("importing numpy as np", flush=True)
    import numpy as np

    task_type = taskTable.get_item(job_id, "TaskType")
    task_type_s3 = task_type_map_s3[task_type]

    model_dir = os.path.join(efs_path, 'model')
    model_bucket = os.environ.get("MODEL_BUCKET")
    model_prefix = "{}/{}".format(task_type_s3, job_id)
    key = "{}/model.tar.gz".format(model_prefix)

    response = s3.get_object(Bucket=model_bucket, Key=key)
    LastModified = response.get("LastModified")
    dt_str = get_datetime_str(LastModified)
    target_dir = os.path.join(model_dir, task_type_s3, f"{job_id}-{dt_str}")
    print(f"target_dir={target_dir}")

    if not os.path.exists(os.path.join(target_dir, "DONE")):
        # if os.path.exists(target_dir):
        #     shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        fullpath = os.path.join(target_dir, "model.tar.gz")

        if not os.path.exists(os.path.join(target_dir, "DOWNLOADED")):
            print("Downloading from s3://{}/{}/model.tar.gz to {}".format(model_bucket, model_prefix, target_dir))
            s3.download_file(model_bucket, key, fullpath)
            np.savetxt(os.path.join(target_dir, "DOWNLOADED"), [])

        if not os.path.exists(os.path.join(target_dir, "EXTRACTED")):
            print("Extracting {} to {}".format(fullpath, target_dir))
            tar = tarfile.open(fullpath)
            tar.extractall(path=target_dir)
            tar.close()
            np.savetxt(os.path.join(target_dir, "EXTRACTED"), [])

        np.savetxt(os.path.join(target_dir, "DONE"), [])

    print("importing mxnet as mx", flush=True)
    import mxnet as mx
    print("importing other libraries...", flush=True)
    import cv2
    from mxnet.gluon.data.vision import transforms
    from mxnet import nd, autograd, gluon
    import gluonnlp as nlp
    # from tokenizers import Tokenizer
    # from tokenizers.models import BPE

    print(mx.__version__)
    print(np.__version__)
    print(nlp.__version__)

    ctx = mx.cpu()

    print(os.listdir(target_dir))
    if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
        model = gluon.nn.SymbolBlock.imports(
            f"{target_dir}/custom_model-symbol.json", ['data'],
            f"{target_dir}/custom_model-0000.params", ctx=ctx)
    elif task_type == "NAMED_ENTITY_RECOGNITION":
        model = gluon.nn.SymbolBlock.imports(
            f"{target_dir}/custom_model-symbol.json", ['data0', 'data1', 'data2'],
            f"{target_dir}/custom_model-0000.params", ctx=ctx)
        with open(f"{target_dir}/text_vocab.json", 'r') as fp:
            text_vocab_str = fp.read()
            text_vocab = nlp.Vocab.from_json(text_vocab_str)
        with open(f"{target_dir}/tag_vocab.json", 'r') as fp:
            tag_vocab_str = fp.read()
            tag_vocab = nlp.Vocab.from_json(tag_vocab_str)
        tokenizer = nlp.data.BERTTokenizer(vocab=text_vocab, lower=True)
    model.hybridize(static_alloc=True, static_shape=True)

    def image_classification_predict(model, input_np):
        """For the input, do the predictions and return them.
        """
        tic = time.time()

        hyperparameters_str = taskTable.get_item(job_id, "HyperParameters")
        if hyperparameters_str is None:
            hyperparameters_str = "{}"
        hyperparameters = ast.literal_eval(hyperparameters_str) # convert string expr to dict
        image_size = int(hyperparameters.get("IMAGE_SIZE", "256"))
        image_size_crop = int(round(image_size * 0.875))
        print(f"image_size={image_size}, image_size_crop={image_size_crop}")
        to_gray = int(hyperparameters.get("TO_GRAY", "0"))
        print(f"to_gray={to_gray}")

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
        
        net = model

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
        with open(f"{target_dir}/classes.txt", "r") as fp:
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

        out = io.StringIO()
        out.write(output_dict_str)
        result = out.getvalue()
        return result

    def object_detection_predict(model, input_np):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        import boto3

        net = model # self.get_model()
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
        
        with open("/mnt/ml/model/classes.txt", "r") as f:
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

    def named_entity_recognition_predict(model, input_np):
        """For the input, do the predictions and return them.
        """
        seq_len = 128
        net = model # self.get_model()

        print(f"input_np={str(input_np)}")

        def encode_as_input(sentence, seq_len, text_vocab, tag_vocab, null_tag_index=0):
            """Enocde a single sentence into numpy arrays as input to the BERTTagger model.
            """
            NULL_TAG = 'X'
            # check whether the given sequence can be fit into `seq_len`.
            # print(''.join([t.text[0] for t in sentence]))
            assert len(sentence) <= seq_len - 2, \
                'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
                .format(len(sentence), seq_len, ' '.join([t[0] for t in sentence]))

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
        sentence = tokenizer(sentence)
        data = encode_as_input(sentence, seq_len, text_vocab, tag_vocab)

        text_ids, token_types, valid_length, tag_ids, _ = data # [x.astype('float32').as_in_context(ctx) for x in data]
        text_ids = text_ids.expand_dims(axis=0).astype('float32').as_in_context(ctx)
        token_types = token_types.expand_dims(axis=0).astype('float32').as_in_context(ctx)
        out = net(text_ids, token_types, valid_length.astype('float32'))

        # convert results to numpy arrays for easier access
        np_text_ids = text_ids.astype('int32').asnumpy().flatten()
        np_pred_tags = out.argmax(axis=-1).asnumpy().flatten()
        np_valid_length = valid_length.astype('int32').asnumpy().flatten()
        np_true_tags = tag_ids.asnumpy().flatten()

        text = [text_vocab.idx_to_token[int(t)] for t in np_text_ids]
        tags = [tag_vocab.idx_to_token[int(t)] for t in np_pred_tags]

        print(text)
        print(tags)

        def token2tag(t):
            if t in ["O", "X"]:
                return t
            else:
                return t[1:].strip("-")
        # labels = list(set([token2tag(token) for token in tag_vocab.idx_to_token]))
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


    if task_type in ["IMAGE_CLASSIFICATION", "OBJECT_DETECTION"]:
        data_bytes = data_json
        headers = {'Content-type': 'image/jpeg'}

        # preprocessing
        data = data_bytes
        print("len(data)={}".format(len(data)))
        data_np = np.fromstring(data, dtype=np.uint8)
        print("data_np.shape={}".format(str(data_np.shape)))
        print(' '.join(['{:x}'.format(d) for d in data_np[:20].tolist()]), flush=True)
        data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
        data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)

    elif task_type in ["NAMED_ENTITY_RECOGNITION"]:
        data_json = json.dumps({"data": data_json})
        headers = {'Content-type': 'application/json'}
        data_bytes = data_json

        # preprocessing
        data = json.loads(data_json)
        data_np = np.asarray(data['data'])

    else:
        headers = {'Content-type': 'application/json'}

    def predict(model, input_np):
        if task_type == "IMAGE_CLASSIFICATION":
            return image_classification_predict(model, input_np)
        elif task_type == "OBJECT_DETECTION":
            return object_detection_predict(model, input_np)
        elif task_type == "NAMED_ENTITY_RECOGNITION":
            return named_entity_recognition_predict(model, input_np)
        else:
            raise RuntimeError("Unknown task type {}".format(task_type))

    response = predict(model, data_np)

    return response
