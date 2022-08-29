from pycocotools.coco import COCO
import numpy as np

T1_CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16', '18', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
T2_CLASS_NAMES = ['35', '36', '37', '38', '39']
T3_CLASS_NAMES = ['40', '41', '42', '43', '44']

TRAIN_CLASS_NAMES = {
    't1': T1_CLASS_NAMES,
    't2': T2_CLASS_NAMES,
    't3': T3_CLASS_NAMES
}

TRAIN_STEP = 't1'

# Train
coco_annotation_file = '/home/henry/work/datasets/toolkit/coco/annotations/toolkit_train.json'
dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/{TRAIN_STEP}_train.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(TRAIN_CLASS_NAMES[TRAIN_STEP]):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created train file')
