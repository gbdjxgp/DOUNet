_base_ = [
    '../_base_/models/BACL_representation.py',
    '../_base_/datasets/dotav1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
optimizer =dict(lr=0.0025,weight_decay=0.00005)
