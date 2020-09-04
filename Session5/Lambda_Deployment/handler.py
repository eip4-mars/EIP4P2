####################################################################################################

try:
    import unzip_requirements
except ImportError:
    pass

import requests
#import cv2
import numpy as np
from PIL import Image

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import numpy as np

import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print("Import End...")

# Define Env Variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-mars-session1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'pose_resnet50_256x256.pt'

#LABELS_PATH=os.environ['LABELS_PATH'] if 'LABELS_PATH' in os.environ else 'sess2_labels.json'

BODY_POINTS = ['r-ankle','r-knee','r-hip','l-hip','l-knee','l-ankle','pelvis','thorax','upper-neck','head-top','r-wrist','r-elbow','r-shoulder','l-shoulder','l-elbow','l-wrist']

CONNECTIONS = [
            ## lower half
            [0,1],
            [1,2],
            [2,6],
            [5,4],
            [4,3],
            [3,6],
            ## Upper Half
            [10,11],
            [11,12],
            [12,8],
            [15,14],
            [14,13],
            [13,8],
            [8,7],
            [8,9],
            ## Connection
            [7,6]
]
THRESHOLD = 0.7
CIRCLE_RADIUS=5

print("Downloading Model...")

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
        
except Exception as e:
    print(repr(e))
    raise(e)


def transform_image(img):
    try:
        transformations = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transformations(img).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(img):
    tensor = transform_image(img=img)
    return model(tensor).detach().numpy()



def hpe(event,context):
    try:
        print(event)
        print(context)
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print("BODY LOADED...")

        ## Read the input image and get Landmarks
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        im = Image.open(io.BytesIO(picture.content)).convert('RGB')
        im_Sq = im.resize((400,400))
        prediction = get_prediction(img = im_Sq)
        joints = get_prediction(im_Sq).squeeze()
        
        # joints = output.detach().numpy()
        in_W,in_H = im.size
        out_W, out_H  = joints.shape[1:]

        joints_centres = []

        for i in range(joints.shape[0]):
            hottest_areas = np.ma.MaskedArray(joints[i], joints[i] < THRESHOLD)
            joint_centre = hottest_areas.nonzero()[0].mean(),hottest_areas.nonzero()[1].mean()
            if np.isnan(joint_centre[1]): joint_centre = np.nan, np.nan
            joints_centres.append(joint_centre)
        print(joints_centres)
        conv_W = lambda x : np.nan if np.isnan(x) else round(x * in_W / out_W)
        conv_H = lambda x : np.nan if np.isnan(x) else round(x * in_H / out_H)

        final_points = [(conv_W(point[1]),conv_H(point[0])) for point in joints_centres]           

        Unidentified_points = sum([1 for pt in final_points if np.isnan(pt[0])])
        print(Unidentified_points)

        if Unidentified_points > 5:
            return {
                "statusCode": 202,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'error': 'Please Try uploading an Image with Human with more than half body visible'})
            }

        draw = ImageDraw.Draw(im)
        for x,y in final_points:
            if not np.isnan(x):
                leftUpPoint = (x-CIRCLE_RADIUS, y-CIRCLE_RADIUS)
                rightDownPoint = (x+CIRCLE_RADIUS, y+CIRCLE_RADIUS)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.ellipse(twoPointList, fill='red')

        for p1,p2 in CONNECTIONS:
            if not np.isnan(final_points[p1][0]) and not np.isnan(final_points[p2][0]):
                start,end = final_points[p1],final_points[p2]
                draw.line([start, end], width = 2)

        del draw
        
        buffered = io.BytesIO()
        im.save(buffered, format="JPEG")
        serialized_img = base64.b64encode(buffered.getvalue()).decode()

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'aligned': serialized_img})
        }
        
        
    
    except Exception as e:
        print(repr(e))
        return {
            "statuscode" : 500,
            "headers" : {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin' : '*',
                'Access-Control-Allow-Credentials' : True
            },
            "body": json.dumps({"error": repr(e)})
        }
