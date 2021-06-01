from coco_dataset import CocoDetection

coco_detection = CocoDetection(root="../data/coco_2017/", annFile="../data/coco_2017/annotations/instances_train2017.json", transform=None, target_transform=None)

print("number of sample: ", coco_detection.__len__())

print("import some sample")
for i in range(2):
    item = 