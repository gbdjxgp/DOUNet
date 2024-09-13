_base_=[
    'LSKNet_Baseline.py'
]
model = dict(
    backbone=dict(
        type='ARCLSKNet',
        init_cfg=dict(type='Pretrained', checkpoint="./data/pretrained/arc_lsknet_t_backbone.pth.tar"),

    )
)
