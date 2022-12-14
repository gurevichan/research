# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    "/home/jovyan/finetune/Segmentation/configs/pascal_voc12.py",
    "/home/jovyan/finetune/Segmentation/configs/default_runtime.py",
    "/home/jovyan/finetune/Segmentation/configs/schedule_40k.py",
]

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
        frozen_stages=4,
    ),
    decode_head=dict(
        type="LinearHead",
        in_channels=2048,
        in_index=3,
        channels=2048,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="whole"),
    init_cfg=dict(type="Pretrained", checkpoint=""),
)
