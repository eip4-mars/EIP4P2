####################################################################################################

try:
    import unzip_requirements
except ImportError:
    pass

import requests
import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc
from PIL import Image

#import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print("Import End...")

# Define Env Variables
#S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-mars-session1'
#MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'Sess2_mobilenetv2_4.pt'

#LABELS_PATH=os.environ['LABELS_PATH'] if 'LABELS_PATH' in os.environ else 'sess2_labels.json'

#print("Downloading Model...")

#s3 = boto3.client('s3')
'''
try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream...")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
        
        lblObj = s3.get_object(Bucket=S3_BUCKET, Key=LABELS_PATH)
        labelsText = lblObj["Body"].read().decode()
        print(labelsText)
        labels = eval(labelsText)
        print(labels)
        
except Exception as e:
    print(repr(e))
    raise(e)

'''



def align_face(event,context):
    try:
        print(event)
        print(context)
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print(body)
        print("BODY LOADED...")
        
        content_type = event.get('headers', {"content-type" : ''}).get('content-type')
        # body = bytes(body,'utf-8')
        multipart_data = decoder.MultipartDecoder(body, content_type).parts[0]
        img_io = io.BytesIO(multipart_data.content)
        img = Image.open(img_io)
        img = cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)

        PREDICTOR_PATH =  "shape_predictor_5_face_landmarks.dat"
        faceDetector = dlib.get_frontal_face_detector()

        landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
        points = fbc.getLandmarks(faceDetector, landmarkDetector, img)

        if len(points) == 0:
            return {
                "statusCode": 202,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'error': 'No Face detected in the image'})
            }

        points = np.array(points)
        img = np.float32(img)/255.0
        
        h = 600
        w = 600

        #Normalize image to output co-ord
        imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), img, points)
        imNorm = np.uint8(imNorm*255)
        
        #encoded_img = base64.b64encode(imNorm).decode("utf-8") 
        serialized_img = base64.b64encode(cv2.imencode('.jpg', imNorm)[1]).decode()
        
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
