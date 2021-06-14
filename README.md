# Object_Detection
This is faster rcnn implementation.
Dataloader and multi gpu training is adopted from pytorch github:
https://github.com/pytorch/vision/tree/master/references/detection

Main difference is I learn to implement layer by my self.


Cmd: 

Single GPU:
```
python -m torch.distributed.launch --nproc_per_node=1  --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --data-path ../data/coco_2017
```

Mutli GPU
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --data-path ../data/coco_2017
```