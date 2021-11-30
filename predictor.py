import sagemaker
from sagemaker.pytorch import PyTorchModel


local_mode = False

if local_mode:
    instance_type = "local"
else:
    instance_type = "ml.m4.xlarge"

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
role = 'arn:aws:iam::507371033430:role/service-role/AmazonSageMaker-ExecutionRole-20211119T082778'

pytorch_model = PyTorchModel(model_data='s3://articles-dataset/models/model_resnet18_dataset2_5.tar.gz',
# pytorch_model = PyTorchModel(model_data='/tmp/model_resnet18_dataset2_5.tar.gz',
                             role=role,
                             entry_point='inference.py',
                             framework_version='1.9.0',
                             py_version='py38')

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type
)
