from pycocotools.coco import COCO
import numpy as np
import random

# Train
coco_annotation_file = '/home/henry/work/datasets/coco/annotations/toolkit_train.json'
dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/all_task_train.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    image_ids.append(image_details['file_name'].split('.')[0])
    cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created train file')

# Val
dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/all_task_val.txt'

random.shuffle(image_ids)
image_ids = image_ids[0:200]
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    if image_details['file_name'].split('.')[0] in image_ids:
        classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created val file')

# Test
coco_annotation_file = '/home/henry/work/datasets/coco/annotations/toolkit_test.json'
dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/all_task_test.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    image_ids.append(image_details['file_name'].split('.')[0])
    cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')
