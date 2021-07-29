# Object_Detection
This is faster rcnn implementation of pytorch.
Dataloader and multi gpu training is adopted from pytorch github:
https://github.com/pytorch/vision/tree/master/references/detection

This would be baseline to build from https://github.com/jwyang/faster-rcnn.pytorch

Cmd: 

Single GPU:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py    --dataset coco --model fasterrcnn_resnet50_fpn --trainable-backbone-layers 4 --epochs 26    --lr-steps 16 22 --aspect-ratio-group-factor 3 --data-path  ../data/coco_2017
```

Mutli GPU
```
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --trainable-backbone-layers 4 --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --data-path ../data/coco_2017
```