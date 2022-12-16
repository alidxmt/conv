
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
import torch
import detectron2
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader

from detectron2 import model_zoo

from predfn import predict
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import importlib
import numpy as np
import pandas as pd
import cv2
import torch

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
import pycocotools.mask as mask_util
from detectron2.engine import DefaultPredictor
from mod_predict import prediction_setup,prediction,get_class_to_category
from foodyai.data.datareg import get_dataframes
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


items = {}
predictor = ''
categories,images,annotations,nutrition = '','','',''



@app.on_event('startup')
async def startup_event():
    items['intr'] = 'I have been activaed!'
    items['model_path'] ='/home/ali_zolfagharian/foodyai/logs/model_final.pth' 
    items['img_base_path'] = '/home/ali_zolfagharian/raw_data/images/'
    items['output_filepath'] = "/home/ali_zolfagharian/foodyai/logs-pred/prediction_test.json"
    items['config_path'] = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    items['model'] = 'model_zoo'
    items['threshold'] = 0.0
    items['predictor'] = prediction_setup(threshold=items['threshold'],model_path=items['model_path'],config_path=items['config_path'],model=items['model'])
    print('--------predictor----------',str(items['predictor']))
    items['class_to_category'] = get_class_to_category()
    items['df'] = [get_dataframes()]
    #categories,images,annotations,nutrition = get_dataframes()
    print('----categories,images,annotations,nutritions----')


#@app.get("/")
#async def root():
#    return {"message": items['intr']}
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    data = {
        "page": "Home page"
    }
    return templates.TemplateResponse("index.html", {"request": request, "data": data})

@app.get("/pred/")
async def pred(id:str):
    return {"message": '|'.join(predict(id=id,p=False))}

@app.get("/predimg/")
async def predimg(path:str):
    return {"message": '|'.join(predict(id=path,p=True))}


@app.get("/predi/")
async def predi(image_path:str):
    pred = items['predictor']
    image_path = '/home/ali_zolfagharian/raw_data/images/'+image_path+'.jpg'
    categories_ids = prediction(predictor=pred,image_path=image_path,class_to_category=items['class_to_category'],output_filepath=items['output_filepath'])['category_id'].tolist()
    cats = []
    for a_id in categories_ids:
        cats.append(items['df'][0][0].loc[items['df'][0][0]['id']==int(a_id)]['name_readable'].to_string(index=False))
    print('----------result-----------')
    #print(items['df'][0][0])
    return {'message': list(dict.fromkeys(cats))}

@app.get("/categories/")
async def categories():
    return {'message':items['df'][0][0].to_csv()}
    #return {'message':items['df'][0][0].to_json()}

@app.get("/nutrition/")
async def nutrition():
    return {'message':items['df'][0][3].to_csv()}
