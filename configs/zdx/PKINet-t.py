_base_=[
    'LSKNet_Baseline.py'
]
checkpoint = 'data/pretrained/pkinet_t_pretrain.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='PKINet',
        arch='T',
        drop_path_rate=0.1,
        out_indices=(1, 2, 3, 4),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        num_outs=5),
)
