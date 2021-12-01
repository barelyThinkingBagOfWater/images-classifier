import json
import os

import boto3

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime = boto3.Session().client('runtime.sagemaker')


def lambda_handler(event, context):
    print("Received payload: " + json.dumps(event, indent=2))

    print(f'calling endpoint:{ENDPOINT_NAME}')
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Accept='application/json',
                                       Body=json.dumps(event, indent=2))
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)

    return result
