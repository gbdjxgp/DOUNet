_base_=[
    'LSKNet_BACL_stage1_exp1.py'
]
model = dict(
    rpn_head=dict(
        type='ARC_OrientedRPNHead',
    )
)
