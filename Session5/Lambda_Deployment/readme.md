## Deployment steps (Ubuntu)

### Requirements: 
- Linux environment
- Serverless framework

    `npm install serverless -g`
- Docker


### Steps
- Upload Files to Lambda and change Model_name/S3_BuckeT in handler.py/serverless.yml accordingly
  - pose_resnet50_256x256.pt    [Link](https://drive.google.com/drive/folders/1IlwfPx3CpLJ44bX8laCaQGYMD7_Nqqpb?usp=sharing)
- Create a serverless Template

    `sls create --template aws-python3 --path <DIR_NAME>`

- go to <DIR_NAME>

- Install python requirements

    `serverless plugin install -n serverless-python-requirements`

- Run this if you get Docker access issue

    `sudo chmod 666 /var/run/docker.sock`

- Replace the following files as per requirement of the service
  - handler.py
  - requirements.txt
  - serverless.yml
  - package.json

- Deploy Model

    `npm run deploy`
    
- Once Deployed Successfully, Go to AWS Console -> API Gateway -> Click on newly created service -> Setting -> Binary Media Types -> Add following
        
    `*/*`
    
    `multipart/form-data`
   ![](https://github.com/eip4-mars/EIP4P2/blob/master/Session5/Lambda_Deployment/multiformdata.gif)
 

