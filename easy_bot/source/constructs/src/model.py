task_type_map = {
    "IMAGE_CLASSIFICATION": "Image Classification",
    "OBJECT_DETECTION": "Object Detection",
    "NAMED_ENTITY_RECOGNITION": "Named Entity Recognition",
    "": "Image Classification",
}
task_type_map_s3 = {k:v.lower().replace(" ", "-") for k, v in task_type_map.items()}
task_type_lut = {v:k for k, v in task_type_map.items()}
