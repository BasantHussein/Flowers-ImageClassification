
import argparse
from torchvision import transforms, datasets, models
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
import torchvision.models as models
import torch.nn as nn
import helper
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import time


def load_and_transform_data(train_dir, valid_dir):

# TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose(
            [transforms.RandomRotation(30),
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),

             ]),

        'valid': transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
             ])}

    # TODO: Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
    image_datasets_valid = datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid'])

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64, shuffle=True)
    return dataloaders_train, dataloaders_valid


# load the pre-trained network
mobilenet = models.mobilenet_v2(pretrained=True, progress=True)


# build a classifier and change the mobilenet classifier with it to make the final model
def model():
    for param in mobilenet.parameters():
        param.requires_grad = False

    classifier = torch.nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1280, 500)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    mobilenet.classifier = classifier
    return mobilenet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mobilenet.to(device)


# define loss function and optimizer
criterion=torch.nn.NLLLoss()
optimizer=torch.optim.Adam(model().classifier.parameters(), lr=0.001)


def validate(dataloaders_valid,model):
    model.to(device)
    valid_loss=0
    accuracy=0
    for i,(images,labels) in enumerate(dataloaders_valid):
        images, labels = images.to(device), labels.to(device)
        output=model.forward(images)
        valid_loss+=criterion(output,labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy


# train the classifier
def train(epochs, print_every, model, dataloaders_train, dataloaders_valid, optimizer, criterion, device):
    model.to(device)

    steps=0
    for e in range(epochs):
        running_loss=0
        model.train()
        for images,labels in dataloaders_train:
            steps+=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output=model.forward(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if steps % print_every==0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate(dataloaders_valid,model)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders_valid)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders_valid)))
                running_loss=0
                model.train()


def save_checkpoint(model, save_dir, input_size, output_size, hidden_units, dropout):
    model.class_to_idx = image_datasets_train.class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_sizes': hidden_units,
                  'dropout':dropout,
                  'state_dict': model.classifier.state_dict(),
                  'arch': mobilenet,
                  'epochs': 8,
                  'learning_rate': 0.001,
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    return checkpoint


#train(epochs, print_every, model, dataloaders_train, dataloaders_valid, optimizer, criterion, device)
#save_checkpoint(model, save_dir, input_size, output_size, hidden_units, dropout)

def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    # Define location of image data
    data_dir = 'F:/basant/DataScientistNanodegree/aipnd-project/Flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    epochs = 8
    print_every = 40
    input_size=1280
    output_size=102
    hidden_units=500
    dropout=0.2


    dataloaders_train, dataloaders_valid = load_and_transform_data(train_dir, valid_dir)

    modell = model()
    if not modell:
        print("Architecture not supported.")
        return


    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(modell.classifier.parameters(), lr=learning_rate)

    train(epochs, print_every, modell, dataloaders_train, dataloaders_valid, optimizer, criterion, device)
    save_checkpoint(modell, save_dir, input_size, output_size, hidden_units, dropout)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an image classifier ',
    )

    parser.add_argument('data_directory', default='F:/basant/DataScientistNanodegree/aipnd-project/Flowers')
    parser.add_argument('--save_dir', default='F:/basant/DataScientistNanodegree/aipnd-project')
    parser.add_argument('--arch', default='mobilenet')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_units', default=500)
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--gpu', action='store_true', default=True)
    input_args = parser.parse_args()

    main(input_args.data_directory, input_args.save_dir, input_args.arch,
         input_args.learning_rate, input_args.hidden_units, input_args.epochs, input_args.gpu)