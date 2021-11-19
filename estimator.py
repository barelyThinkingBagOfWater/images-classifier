import sagemaker
from sagemaker.pytorch import PyTorch

if __name__ == '__main__':
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    prefix = "dataset2"
    role = 'arn:aws:iam::507371033430:role/service-role/AmazonSageMaker-ExecutionRole-20211119T082778'

    pytorch_estimator = PyTorch('pytorch-train.py',
                                role=role,
                                instance_type='ml.m5.xlarge',
                                instance_count=1,
                                framework_version='1.9.0',
                                py_version='py38',
                                hyperparameters={'epochs': 1,
                                                 'model-dir': 's3://articles-dataset/models',
                                                 'output-data-dir': 's3://articles-dataset/data_output'})

    pytorch_estimator.fit({'train': 's3://articles-dataset/dataset2/train',
                           'test': 's3://articles-dataset/dataset2/val',
                           'control': 's3://articles-dataset/dataset2/control'})
