# image-classifier

In the root folder you'll find a Flask service that will download a model .pth from S3 and use it to classify sent images.
To run it locally create an .env file with the following content:
```
MODEL_BUCKET=S3_BUCKET
MODEL_PATH=models/model.pth
PREDICTION_CLASS_0=categorie_0
PREDICTION_CLASS_1=categorie_1
```


in the sagemaker folder you'll find scripts to interact with AWS Sagemaker to train models remotely using different parameters
and to deploy the models to a SageMaker endpoint with a Lambda function to interact with this endpoint as it's not public
(you have to use an API gateway to call the lambda function)
