Group Members : Deepak Gowtham, Bikash Bhoi

### Serverless MobilenetV2 deployment on AWS Lambda

#### Stpes:
-  Create Linux environemnt
-  Install 
  -  Docker
  - Node.js
  - Serverless Framework
  - Python requirements Plugin (`serverless-python-requirements`)
- Create IAM user and set up SLS with the user credentials
- Create SLS Template
- Create a s3 Bucket and Upload the pretrained model to the Bucket
- Change 
  - handler.py
  - serverless.yml
  - requirements.txt
  - package.json
- Deploy using npm
- Get the Endpoint and call the API 

#### Input Image:
![input](https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2019/12/03202400/Yellow-Labrador-Retriever.jpg)

#### Output Screenshot:
![screenshot](https://github.com/eip4-mars/EIP4P2/blob/master/Session1/Postman_sess1.png)
