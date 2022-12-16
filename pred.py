


print('---libraries---')
import os
import json
import importlib
import numpy as np
import cv2
import torch
import detectron2
from detectron2.engine import DefaultPredictor

#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
import pycocotools.mask as mask_util
from detectron2 import model_zoo
#from detectron2.utils.logger import setup_logger
#setup_logger()




print('---paths---')


#cfg= get_cfg()
threshold = 0.0
#cfg.OUTPUT_DIR = "logs/"
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
model_path = '/home/ali_zolfagharian/foodyai/logs/model_final.pth'#os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
img_path = "/home/ali_zolfagharian/raw_data/images/055902.jpg"
output_filepath = "/home/ali_zolfagharian/foodyai/logs-pred/prediction_test.json"


print('---getting model---')

model_name = "model_zoo"
model = importlib.import_module(f"detectron2.{model_name}")

cfg = get_cfg()
cfg.merge_from_file(model.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_path 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 323

cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

with open("/home/ali_zolfagharian/foodyai/class_to_category.json") as fp:
        class_to_category = json.load(fp)



img = cv2.imread(img_path)
prediction = predictor(img)






annotations = []
instances = prediction["instances"]
print(instances)
print(len(instances))
if len(instances)==0: print('! -----------the instances is empty')
if len(instances) > 0:
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    bboxes = BoxMode.convert(
        instances.pred_boxes.tensor.cpu(),
        BoxMode.XYXY_ABS,
        BoxMode.XYWH_ABS,
    ).tolist()

    masks = []
    if instances.has("pred_masks"):
        for mask in instances.pred_masks.cpu():
            _mask = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            _mask["counts"] = _mask["counts"].decode("utf-8")
            masks.append(_mask)

    for idx in range(len(instances)):
        category_id = class_to_category[str(classes[idx])]
        output = {
            "image_id": int(os.path.basename(img_path).split(".")[0]),
            "category_id": category_id,
            "bbox": bboxes[idx],
            "score": scores[idx],
        }
        #print(output)
        if len(masks) > 0:
            output["segmentation"] = masks[idx]
        annotations.append(output)

print('opening prediction file')

with open(output_filepath, "w") as fp:
    json.dump(annotations, fp)


f = open(output_filepath)
pred = json.load(f)

from foodyai.data.datareg import get_dataframes
a,b,c,d = get_dataframes()

name_list = []
for i in range(len(pred)):
	id =(pred[i]['category_id'])
	name_list.append((a[a['id']==id]['name']).tolist()[0])

name_list = list(dict.fromkeys(name_list))

print('--------categories-----')
print(f'number of categories is {len(name_list)}')
for name in name_list:
      print(name)

print('end')

