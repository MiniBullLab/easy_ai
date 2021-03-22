# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
try:
    from easyai import _C
    try:
        from apex import amp
        # Only valid with fp32 inputs - give AMP the hint
        nms = amp.float_function(_C.nms)
    except ImportError:
        nms = _C.nms
        print("import amp fail!")
except ImportError:
    nms = None
    print("import _C fail!")

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""