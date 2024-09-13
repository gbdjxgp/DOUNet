_base_ = [
    '../_base_/models/BACL_classifier.py',
    '../_base_/datasets/dotav1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
load_from = 'work_dirs/BACL_representation_dota10/epoch_12.pth'