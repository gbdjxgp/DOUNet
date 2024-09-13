_base_=[
    'LSKNet_Baseline.py'
]
model = dict(
    backbone=dict(
        type='ARCLSKNet',
        init_cfg=None
    )
)
