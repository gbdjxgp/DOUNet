_base_ = [
    'LSKNet_Baseline.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
            ),
        )
    )
)