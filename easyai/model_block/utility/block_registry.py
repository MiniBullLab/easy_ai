from easyai.utility.registry import Registry

# backbone
REGISTERED_CLS_BACKBONE = Registry("cls_model")

REGISTERED_VISION_BACKBONE = Registry("torch_vision_model")

REGISTERED_GAN_D_BACKBONE = Registry("gan_d_model")
REGISTERED_GAN_G_BACKBONE = Registry("gan_G_model")

REGISTERED_PC_CLS_BACKBONE = Registry("pc_cls_model")

# head
REGISTERED_MODEL_HEAD = Registry("model_head")

# neck
REGISTERED_MODEL_NECK = Registry("model_neck")

