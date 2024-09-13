_base_ = [
    'OrientedRCNN.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='BCE', use_sigmoid=True, loss_weight=1.0,num_classes=15
            )
        )
    )
)
