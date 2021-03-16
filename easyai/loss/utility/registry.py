from easyai.utility.registry import Registry

REGISTERED_COMMON_LOSS = Registry("common_loss")
REGISTERED_CLS_LOSS = Registry("cls_loss")
REGISTERED_DET2D_LOSS = Registry("det2d_loss")
REGISTERED_SEG_LOSS = Registry("seg_loss")
REGISTERED_DET3D_LOSS = Registry("det3d_loss")
REGISTERED_KEYPOINT2D_LOSS = Registry("keypoint2d_loss")
REGISTERED_POSE2D_LOSS = Registry("pose2d_loss")

REGISTERED_GAN_D_LOSS = Registry("gan_d_loss")
REGISTERED_GAN_G_LOSS = Registry("gan_g_loss")
