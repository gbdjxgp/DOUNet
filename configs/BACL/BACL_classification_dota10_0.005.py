_base_ = [
    '../_base_/models/BACL_classifier.py',
    '../_base_/datasets/dotav1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
optimizer =dict(lr=0.005)
load_from = 'work_dirs/BACL_representation_dota10_0.005/epoch_12.pth'