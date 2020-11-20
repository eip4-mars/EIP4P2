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

import pickle
import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder

from models import EncoderCNN, DecoderRNN

print("Import End...")


# Define Env Variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-mars-session1'

ENC_PATH = os.environ['ENC_PATH'] if 'ENC_PATH' in os.environ else 'encoderdata.pkl'
DEC_PATH = os.environ['DEC_PATH'] if 'DEC_PATH' in os.environ else 'decoderdata.pkl'
VOCAB_PATH = os.environ['VOCAB_PATH'] if 'VOCAB_PATH' in os.environ else 'decoder_vocab.pickle'
DEC_PARAM = os.environ['DEC_PARAM'] if 'DEC_PARAM' in os.environ else 'decoder_input_params.pickle'

print("Downloading Model...")

s3 = boto3.client('s3')
DEVICE = torch.device('cpu')

model_save_path = '/tmp/'

try:
    # print(VOCAB_PATH + ' Download')
    # s3.download_file(S3_BUCKET, VOCAB_PATH, model_save_path+VOCAB_PATH)
    # s3.download_file(S3_BUCKET, DEC_PARAM, model_save_path+DEC_PARAM)
    # print(DEC_PARAM + 'Downloaded.')
    # print(ENC_PATH + ' Download.')
    # s3.download_file(S3_BUCKET, ENC_PATH, model_save_path+ENC_PATH)
    # print(DEC_PATH + 'Download.')
    # s3.download_file(S3_BUCKET, DEC_PATH, model_save_path+DEC_PATH)

    
    # os.makedirs( model_save_path , exist_ok=True)


    # with open(  os.path.join(model_save_path , 'decoder_input_params.pickle'), 'rb') as handle:
    #     decoder_input_params = pickle.load(handle)
    # with open(  os.path.join(model_save_path , 'decoder_vocab.pickle'), 'rb') as handle:
    #     decoder_vocab = pickle.load(handle)

    ## Load Decoder Params
    Obj = s3.get_object(Bucket=S3_BUCKET, Key=DEC_PARAM)
    bytestream = io.BytesIO(Obj['Body'].read())
    decoder_input_params = pickle.load(bytestream)
    print(decoder_input_params)

    embed_size = decoder_input_params['embed_size']
    hidden_size= decoder_input_params['hidden_size']
    vocab_size = decoder_input_params['vocab_size']
    num_layers = decoder_input_params['num_layers']

    ## Load Vocab
    Obj = s3.get_object(Bucket=S3_BUCKET, Key=VOCAB_PATH)
    bytestream = io.BytesIO(Obj['Body'].read())
    decoder_vocab = pickle.load(bytestream)
    print('decoder_vocab loaded')

    # Load Encoder
    Obj2 = s3.get_object(Bucket=S3_BUCKET, Key=ENC_PATH)
    bytestream = io.BytesIO(Obj2['Body'].read())
    encoder_model = EncoderCNN(embed_size)
    encoder_model.load_state_dict(torch.load(bytestream, map_location = DEVICE))
    print('Encoder loaded')

    # Load Decoder
    Obj3 = s3.get_object(Bucket=S3_BUCKET, Key=DEC_PATH)
    bytestream = io.BytesIO(Obj3['Body'].read())
    decoder_model = DecoderRNN( embed_size , hidden_size , vocab_size , num_layers )
    decoder_model.load_state_dict(torch.load(bytestream, map_location = DEVICE))
    print('Decoder loaded')

    # decoder = DecoderRNN( embed_size , hidden_size , vocab_size , num_layers )
    # decoder.load_state_dict( torch.load(   os.path.join( model_save_path , 'decoderdata.pkl' )   ) )
    encoder_model.eval()
    decoder_model.eval()

except Exception as e:
    print('error in loading block')
    print(repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        transform_test = transforms.Compose([ 
                         transforms.Resize(224),                          # smaller edge of image resized to 256
                         transforms.RandomCrop(224),                      # get 224x224 crop from random location
                         transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                         transforms.ToTensor(),                           # convert the PIL Image to a tensor
                         transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                             (0.229, 0.224, 0.225))])
        image = Image.open(io.BytesIO(image_bytes))
        return transform_test(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_sentences(all_predictions ):
    return ' '.join([decoder_vocab[idx] for idx in all_predictions[1:-1] ]  )

def get_caption(image_bytes):
    print('applying transform')
    tensor = transform_image(image_bytes=image_bytes)
    print('Encoding...')
    features  = encoder_model(tensor).unsqueeze(1)
    print('Decoding...')
    final_output = decoder_model.Predict( features, max_len=20)
    return get_sentences(final_output)

def caption(event,context):
    try:
        print(event)
        print(context)
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print("BODY LOADED...")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        caption = get_caption(image_bytes = picture.content)
        
        # caption ='default caption'
        print(caption)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'Caption': caption})
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
