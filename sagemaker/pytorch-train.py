from __future__ import print_function, division

import argparse
import copy
import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import boto3


def transform_model(train_dataset_path, test_dataset_path, epochs, learning_rate):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # Do we want some data augmentation for our set?
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(train_dataset_path), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(test_dataset_path), data_transforms['val'])}
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=2),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=True, num_workers=2)}
    dataset_sizes = {'train': len(image_datasets['train']),
                     'val': len(image_datasets['val'])}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    # model = torchvision.models.resnet152(pretrained=True)
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=epochs, dataloaders=dataloaders,
                       device=device, dataset_sizes=dataset_sizes), class_names


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, device, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def check_model_with_control_data(model, control_dataset_path, class_names):
    # Now to test the model with 5% of pictures of each category
    print("")
    print("Now running control phase")
    model.eval()

    # Same as the validation set, just normalization for validation
    transform_control_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    error_count = 0
    total_count = 0
    for class_name in class_names:
        for picture in os.listdir(control_dataset_path + "/" + class_name):
            img = Image.open(control_dataset_path + "/" + class_name + "/" + picture)
            x = transform_control_image(img)  # Preprocess image
            x = x.unsqueeze(0)  # Add batch dimension
            total_count = total_count + 1

            output = model(x)  # Forward pass
            pred = torch.argmax(output, 1)  # Get predicted class

            status = 'SUCCESS:'
            if not class_name == class_names[pred]:
                status = 'FAILURE:'
                error_count = error_count + 1

            print(status + " " + class_name + ' image named:' + picture +
                  ' predicted as', class_names[pred])

    success_rate: float = round((float(error_count) / total_count) * 100, 2)

    print("")
    print("Results with the control data: " + str(error_count) +
          " misclassifications on " + str(total_count) +
          " predictions or an error rate of " + str(success_rate) + "%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.001, dest='learning_rate')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'], dest='output_data_dir')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'], dest='model_dir')
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--control', type=str, default=os.environ['SM_CHANNEL_CONTROL'])

    args, _ = parser.parse_known_args()

    print("Using train dataset:" + str(args.train))
    print("Using number of epochs:" + str(args.epochs))
    print("Using learning rate:" + str(args.learning_rate))
    print("Using output-data-dir:" + args.output_data_dir)

    instance_type = "local"
    try:
        if subprocess.call("nvidia-smi") == 0:
            instance_type = "local_gpu"
    except:
        pass

    model, class_names = transform_model(args.train, args.test, args.epochs, args.learning_rate)

    check_model_with_control_data(model, args.control, class_names)

    # Change it depending on the pretrained model
    model_file_name = 'model_resnet18_dataset2_' + str(args.epochs) + '.pth'
    temp_model_path = "/opt/ml/output/"
    model_local_path = os.path.join(temp_model_path, model_file_name)
    torch.save(model.state_dict(), model_local_path)

    s3_client = boto3.client('s3')
    s3_client.upload_file(os.path.join(temp_model_path, model_file_name), "articles-dataset", "models/" +
                          model_file_name)