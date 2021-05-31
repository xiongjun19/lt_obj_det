# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation

# Modify dataset related settings
dataset_type = 'COCODataset'
train_prefix = "/home/ubuntu/data/coco/train2017"
val_prefix = "/home/ubuntu/data/coco/val2017"
data = dict(
    train=dict(
        img_prefix=train_prefix,
        ann_file='/home/ubuntu/data/coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix=val_prefix,
        ann_file='/home/ubuntu/data/coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix=val_prefix,
        ann_file='/home/ubuntu/data/coco/annotations/instances_val2017.json'))

