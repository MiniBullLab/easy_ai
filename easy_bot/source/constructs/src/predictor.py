import io
import os
import sys
import time
import boto3
import tarfile
import codecs
import json
import ast
from dynamodb_controller import TaskTable
from model import task_type_map, task_type_map_s3, task_type_lut
from utils import get_datetime_str

import cv2
from easyai.tools.task_tool.bot_inference import BotInference

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

    print(os.listdir(target_dir))

    if task_type == "IMAGE_CLASSIFICATION":
        model = BotInference("classify", 1)
        model_path = os.path.join(target_dir, "model/cls_best.pt")
        config_path = os.path.join(target_dir, "model/classify_config.json")
        model.build_task("classnet", 0, model_path, config_path)
    elif task_type == "OBJECT_DETECTION":
        model = BotInference("detect2d", 1)
        model_path = os.path.join(target_dir, "model/det2d_best.pt")
        config_path = os.path.join(target_dir, "model/detection2d_config.json")
        model.build_task("denet", 0, model_path, config_path)
    elif task_type == "NAMED_ENTITY_RECOGNITION":
        model = None

    def image_classification_predict(model, input_np):
        """For the input, do the predictions and return them.
        """
        tic = time.time()

        hyperparameters_str = taskTable.get_item(job_id, "HyperParameters")
        if hyperparameters_str is None:
            hyperparameters_str = "{}"
        hyperparameters = ast.literal_eval(hyperparameters_str) # convert string expr to dict

        net = model

        toc = time.time()
        print(f"0.1 - elapsed: {(toc-tic)*1000.0} ms")

        class_index, class_confidence = net.infer(input_np)

        toc = time.time()
        print(f"0.3 - elapsed: {(toc-tic)*1000.0} ms")

        cls_config_path = os.path.join(target_dir, "model/classify_config.json")
        if os.path.exists(cls_config_path):
            with codecs.open(cls_config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            return None

        temp_results = [{"Class": config_dict['class_name'][class_index],
                        "Probability": class_confidence}]
        print("results={}".format(str(temp_results)))
        output_dict = {"ClassIndex": class_index, "Results": temp_results}
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
        net = model
        detection_objects, _ = net.infer(input_np)

        det2d_config_path = os.path.join(target_dir, "model/detection2d_config.json")
        if os.path.exists(det2d_config_path):
            with codecs.open(det2d_config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
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

    def named_entity_recognition_predict(model, input_np):
        results = {
            "Labels": [],
            "Results": [],
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
