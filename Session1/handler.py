#import json
#
#
#def hello(event, context):
#    body = {
#        "message": "Go Serverless v1.0! Your function executed successfully!",
#        "input": event
#    }
#
#    response = {
#        "statusCode": 200,
#        "body": json.dumps(body)
#    }
#
#    return response
#
#    # Use this code if you don't use the http event with the LAMBDA-PROXY
#    # integration
#    """
#    return {
#        "message": "Go Serverless v1.0! Your function executed successfully!",
#        "event": event
#    }
#    """
#
#
####################################################################################################

try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import requests

import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print("Import End...")


# Define Env Variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-mars-session1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenetV2.pt'

LABELS_PATH=os.environ['LABELS_PATH'] if 'LABELS_PATH' in os.environ else 'imagenet_labels.json'

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
        
        lblObj = s3.get_object(Bucket=S3_BUCKET, Key=LABELS_PATH)
        labelsText = lblObj["Body"].read().decode()
        print(labelsText)
        labels = eval(labelsText)
        print(labels)
        
except Exception as e:
    print(repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()



def classify_image(event,context):
    try:
        print(event)
        print(context)
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print("BODY LOADED...")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes = picture.content)
        label = labels[str(prediction)]

        print(prediction,label)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print('filename: '+filename.replace('"',''))

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction, 'label': label})
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
