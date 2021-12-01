import boto3
import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
import os


def init_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Downloading and loading the model.')

    model = models.resnet152(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model_local_file = "model.pth"
    model_bucket = os.environ.get("MODEL_BUCKET")
    model_path = os.environ.get("MODEL_PATH")
    s3_client = boto3.client('s3')
    s3_client.download_file(model_bucket, model_path, model_local_file)
    print("Model downloaded")

    with open(model_local_file, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    print('The model is now loaded')
    return model


class PredictionService:
    def __init__(self):
        self.model = init_model()

    def predict_image(self, image: Image):
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data = image_transform(image)
        classes = {0: os.environ.get("PREDICTION_CLASS_0"), 1: os.environ.get("PREDICTION_CLASS_1")}

        if torch.cuda.is_available():
            input_data = data.view(1, 3, 224, 224).cuda()
        else:
            input_data = data.view(1, 3, 224, 224)

        with torch.no_grad():
            self.model.eval()
            out = self.model(input_data)

            prediction = np.argmax(out)
            predicted_class = classes.get(prediction.item())
            print(f'predicted class:{predicted_class}')
            return predicted_class
