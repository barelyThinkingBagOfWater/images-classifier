import json
import requests
import boto3
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import transforms


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading the model.')
    model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model_local_file = "/opt/ml/input/model.pth"
    s3_client = boto3.client('s3')
    s3_client.download_file("articles-dataset", "models/model_resnet18_dataset2_5.pth", model_local_file)

    with open(model_local_file, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()

    print('Done loading model')
    return model


def input_fn(request_body, request_content_type):
    input_data = json.loads(request_body)
    url = input_data['url']
    print(f'Image url: {url}')

    image_data = Image.open(requests.get(url, stream=True).raw)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return image_transform(image_data)


def predict_fn(input_data, model):
    print('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)

    return ps


def output_fn(prediction_output, content_type):
    print('Serializing the generated output.')
    classes = {0: 'Bees', 1: 'Ants'}

    topk, topclass = prediction_output.topk(3, dim=1)
    pred = {'prediction': classes[topclass.cpu().numpy()[0]],
            'score': f'{topk.cpu().numpy()[0] * 100}%'}
    return json.dumps(pred)
