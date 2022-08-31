import math

import cv2
import os
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    try:
        pdf = distribution.log_prob(dx).exp()
    except ValueError:
        pdf = float('nan')
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    cls = classes
    lse = torch.logsumexp(logits[:, :cfg.OWOD.CUR_INTRODUCED_CLS], dim=1)
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)
        # print(str(p_unk) + '  --  ' + str(p_known))
        if torch.isnan(p_unk) or torch.isnan(p_known):
            continue
        if p_unk > p_known:
            cls[i] = unknown_class_index
    return cls


# Get image
datasets_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
file_name = "18-57-24"
im = cv2.imread(os.path.join(datasets_root, 'VOC2007/JPEGImages', file_name + '.jpg'))

# model = './output/t2_ft/model_final.pth'
# model = './output/t3_ft/model_final.pth'
model = './output/t1/model_final.pth'
cfg_file = './configs/OWOD_TOOLKIT/t1/t1_test.yaml'

# Get the configuration ready
cfg = get_cfg()
cfg.merge_from_file(cfg_file)
cfg.MODEL.WEIGHTS = model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print('Before' + str(outputs['instances'].pred_classes))

params = torch.load(os.path.join('./output/t1_final/energy_dist_' + str(21) + '.pkl'))
unknown = params[0]
known = params[1]
unk_dist = create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
known_dist = create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])

instances = outputs['instances'].to(torch.device('cpu'))
dev = instances.pred_classes.get_device()
classes = instances.pred_classes.tolist()
classes = update_label_based_on_energy(instances.logits, classes, unk_dist, known_dist)
classes = torch.IntTensor(classes).to(torch.device('cuda'))
outputs['instances'].pred_classes = classes
print(classes)
print('After' + str(outputs['instances'].pred_classes))


v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img = v.get_image()[:, :, ::-1]
cv2.imwrite(os.path.join('./output/' + file_name + '.jpg'), img)

