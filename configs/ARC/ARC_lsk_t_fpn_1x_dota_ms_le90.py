_base_ = [
    '../lsknet/lsk_t_fpn_1x_dota_ms_le90.py'
]
model = dict(
    # pretrained='pretrained/ARC_ResNet50_xFFF_n4.pth',
    backbone=dict(
        _delete_=True,
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/ARC_ResNet50_xFFF_n4.pth'),
        type='ARCResNet',
        depth=50,

        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',

        replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5'],
            ['0', '1', '2']
        ],
        kernel_number = 4,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5)
        }
    )
)