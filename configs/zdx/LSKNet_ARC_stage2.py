_base_=[
    'LSKNet_ARC.py'
]
model = dict(
    rpn_head=dict(
        type='ARC_OrientedRPNHead',
    ),
    roi_head=dict(
        type='FhmOrientedStandardRoIHead',
        bbox_head=dict(
            type='RotatedFhmShared2FCBBoxHead',
            loss_cls=dict(
                type='FCBL', use_sigmoid=True, loss_weight=1.0,num_classes=15, alpha=0.85, prob_thr=0.7
            ),
            fhm_cfg=dict(
                decay_ratio=0.1,
                sampled_num_classes=8,
                sampled_num_features=12
            )
        )
    ),
)
custom_hooks = [
    dict(
        type="ReweightHook",
        step=1)
]
load_from = 'work_dirs/LSKNet_ARC/epoch_12.pth'
selectp = 1