import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import transforms


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading the model.')
    model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model_local_file = "/tmp/model.pth"

    with open(model_local_file, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()

    print('Done loading model')
    return model


if __name__ == '__main__':
    model = load_model()
    ant_image_url1 = "https://i.ibb.co/sbSBKWq/2238242353-52c82441df.jpg"
    ant_image_url2 = "https://i.ibb.co/55XSpWf/Hormiga.jpg"
    bee_image_url1 = "https://i.ibb.co/0hhCHGs/2745389517-250a397f31.jpg"
    bee_image_url2 = "https://i.ibb.co/Pm6m17q/2883093452-7e3a1eb53f.jpg"

    image_data = Image.open(requests.get(ant_image_url1, stream=True).raw)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = image_transform(image_data)
    classes = {0: 'Ant', 1: 'Bee'}

    print('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = data.view(1, 3, 224, 224).cuda()
    else:
        input_data = data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(input_data)

        predicted_class = np.argmax(out)
        print(f'predicted class:{classes.get(predicted_class.item())}')

