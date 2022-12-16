



import torch, detectron2

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog, DatasetCatalog

from pycocotools.coco import COCO

from detectron2.data.datasets import register_coco_instances


train_annotations_path = '/home/ali_zolfagharian/raw_data/annotations.json'
train_images_path = '/home/ali_zolfagharian/raw_data/images'

val_annotations_path = '/home/ali_zolfagharian/raw_data/annotations.json'
val_images_path = '/home/ali_zolfagharian/raw_data/images'

train_coco = COCO(train_annotations_path)

register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)

print("Registered coco instances")


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# Check the model zoo and use any of the models ( from detectron2 github repo)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))


cfg.DATASETS.TRAIN = ("training_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2
# Loading pre trained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# No. of Batchs
cfg.SOLVER.IMS_PER_BATCH = 10   # This is the real "batch size" commonly known to deep learning people

# Learning Rate: 
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR

# No of Interations
cfg.SOLVER.MAX_ITER = 100 # Try with less to start with

# Images per batch (Batch Size) 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

# No of Categories(Classes) present
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 323

cfg.OUTPUT_DIR = "logs/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

print('trained')



