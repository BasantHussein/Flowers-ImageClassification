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
import json
import tensorflow as tf
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.mobilenet_v2(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier =torch.nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'] ,checkpoint['hidden_sizes'])),
        ('relu' ,nn.ReLU()),
        ('dropout' ,nn.Dropout(checkpoint['dropout'])),
        ('fc2', nn.Linear(checkpoint['hidden_sizes'] ,checkpoint['output_size'])),
        ('output' ,nn.LogSoftmax(dim=1))
    ]))


   # model = nn.DataParallel(model)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    classifier.load_state_dict(checkpoint['state_dict'])

    model.classifier = classifier
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epochs']


    return model, optimizer, start_epoch


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)

    # Resize
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(height * float(new_width) / width)
    else:
        new_height = 256
        new_width = int(width * float(new_height) / height)

    image = image.resize((new_width, new_height))

    # Crop
    left = (new_width - 224) / 2
    right = new_width - (new_width - 224) / 2
    upper = (new_height - 224) / 2
    lower = new_height - (new_height - 224) / 2
    image = image.crop((left, upper, right, lower))

    # Normalize
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image) / 255.0
    np_image = (np_image - means) / stds

    # Transpose
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    #model = tf.convert_to_tensor(model)
    #model=torch.tensor(model).to(device)
    model.to(device)
    model.eval()

    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probs, labels = output.topk(topk)
    probs = np.array(probs.exp().data)[0]
    labels = np.array(labels)[0]

    return probs, labels


def main(image_path, checkpoint, top_k, category_names, gpu):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

   # checkpoint=torch.load(checkpoint)
    model = load_checkpoint(checkpoint)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    probs, classes = predict(image_path, model, device, top_k)

    y_pos = np.arange(len(classes))
    labels = []
    for cls in classes:
        labels.append(cat_to_name[cls])

    return labels, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict an image class ',
    )

    parser.add_argument('image_path', default='F:/basant/DataScientistNanodegree/aipnd-project/Flowers/train/1/image_06734.jpg')
    parser.add_argument('checkpoint', default='F:/basant/DataScientistNanodegree/aipnd-project/checkpoint.pth')
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--category_names', default='F:/basant/DataScientistNanodegree/aipnd-project/cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    input_args = parser.parse_args()

    classes, probs = main(input_args.image_path, input_args.checkpoint, input_args.top_k,
                          input_args.category_names, input_args.gpu)

    print([x for x in zip(classes, probs)])


