import os
import io
import time
import ast

import cv2
import warnings
import numpy as np

import codecs
import json
import traceback
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("ai_runtime.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")
from easyai.tools.task_tool.bot_inference import BotInference

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
        pass

    @classmethod
    def predict(self, input_np):
        if self.task_type == "IMAGE_CLASSIFICATION":
            return self.image_classification_predict(input_np)
        elif self.task_type == "OBJECT_DETECTION":
            return self.object_detection_predict(input_np)
        elif self.task_type == "NAMED_ENTITY_RECOGNITION":
            return self.segment_predict(input_np)
        else:
            raise RuntimeError("Unknown task type {}".format(self.task_type))

    @classmethod
    def image_classification_predict(cls, input_np):
        tic = time.time()
        if os.path.exists("/opt/ml/model/classify_config.json"):
            with codecs.open("/opt/ml/model/classify_config.json", 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            EasyLogger.error("/opt/ml/model/classify_config.json not exits")
            return None
        try:
            inference_task = BotInference("classify", 1)
            inference_task.build_task("classnet", 0,
                                      "/opt/ml/model/cls_best.pt",
                                      "/opt/ml/model/classify_config.json")
            class_index, class_confidence = inference_task.infer(input_np)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
            return None

        results = [{"Class": config_dict['class_name'][class_index],
                    "Probability": class_confidence}]
        print("results={}".format(str(results)))
        output_dict = {"ClassIndex": class_index, "Results": results}
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
    def object_detection_predict(cls, input_np):
        if os.path.exists("/opt/ml/model/detection2d_config.json"):
            with codecs.open("/opt/ml/model/detection2d_config.json", 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            EasyLogger.error("/opt/ml/model/detection2d_config.json not exits")
            return None
        try:
            inference_task = BotInference("detect2d", 1)
            inference_task.build_task("denet", 0,
                                      "/opt/ml/model/det2d_best.pt",
                                      "/opt/ml/model/detection2d_config.json")
            detection_objects, _ = inference_task.infer(input_np)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
            return None

        points_result = []
        id_result = []
        score_result = []
        for object in detection_objects:
            corner_points = [int(object.min_corner.x),
                             int(object.min_corner.y),
                             int(object.max_corner.x),
                             int(object.max_corner.y)]
            points_result.append(corner_points)
            id_result.append(int(object.classIndex))
            score_result.append(float(object.classConfidence))

        result_dict = {"Result":
                           {"class_IDs": id_result,
                            "classes": config_dict['detect2d_class'],
                            "threshold": 0.5,
                            "scores": score_result,
                            "bounding_boxs": points_result
                            },
                       "height": input_np.shape[0],
                       "width": input_np.shape[1]
                       }
        result_str = json.dumps(result_dict)
        print("object detection success")
        return result_str

    @classmethod
    def segment_predict(cls, input_np):
        results = {
            "Labels": [],
            "Results": [],
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
