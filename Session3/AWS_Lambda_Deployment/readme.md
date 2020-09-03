## Deployment steps (Ubuntu)

### Requirements: 
- Linux environment
- Serverless framework

    `npm install serverless -g`
- Docker


### Steps
- Upload Files to Lambda and change Model_name/S3_BuckeT in handler.py/serverless.yml accordingly
  - Sess2_mobilenetv2_4.pt    [Link for both](https://drive.google.com/drive/folders/19oJh4ZgDcSqV0FzK0XD2Y10EFGcD5GCm?usp=sharing)
  - sess2_labels.json
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
  - faceBlendCommon.py

- Deploy Model

    `npm run deploy`
    
    
    
 

