import itertools
import random
from pycocotools.coco import COCO

T1_CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                  '18', '19', '20']
T2_CLASS_NAMES = ['21', '22', '23', '24', '25']
T3_CLASS_NAMES = ['26', '27', '28', '29', '30']
OTHER_CLASS_NAMES = ['31', '32', '33', '34']

FT_CLASS_NAMES = {
    't2': list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES)),
    't3': list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES)),
    'val': list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, OTHER_CLASS_NAMES))
}

# change per stage
TRAIN_STEP = 't3'

known_classes = FT_CLASS_NAMES[TRAIN_STEP]
coco_annotation_file = '/home/henry/work/datasets/coco/annotations/toolkit_train.json'
items_per_class = 20

if TRAIN_STEP == 'val':
    dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/all_task_val.txt'  # IN EXPERIMENT
else:
    dest_file = f'/home/henry/work/OWOD/datasets/OWOD_toolkit/{TRAIN_STEP}_ft.txt'


coco_instance = COCO(coco_annotation_file)
file_names = []

for index, cat_id in enumerate(coco_instance.catToImgs):
    cat_details = coco_instance.cats[cat_id]
    if cat_details['name'] in known_classes:
        image_ids = list(set(coco_instance.catToImgs[cat_id]))
        random.shuffle(image_ids)
        for image_id in image_ids[0:items_per_class]:
            file_names.append(coco_instance.imgs[image_id]['file_name'].split('.')[0])

print(len(file_names))
print(len(set(file_names)))

filtered_file_names = set(file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
