from easyai.utility.registry import Registry

REGISTERED_DATASET = Registry("dataset")

REGISTERED_DATASET_COLLATE = Registry("dataset_collate")

REGISTERED_TRAIN_DATALOADER = Registry("train_dataloader")
REGISTERED_VAL_DATALOADER = Registry("val_dataloader")

REGISTERED_DATA_TRANSFORMS = Registry("data_transforms")

REGISTERED_BATCH_DATA_PROCESS = Registry("batch_data_process")


