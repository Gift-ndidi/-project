#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'
import pandas as pd
import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_arg = get_input_args()
    
    model = load_checkpoint(in_arg.checkpoint_path)
    model.eval()
    
    probs, classes = predict(in_arg.image, model, topk=5)
    
    flower_name = flower_names(classes, model.class_to_idx, in_arg.name_flower)
    
    # Result of Prediction
    
    result = pd.DataFrame({'flower': flower_names, 'probability': probs})
    result = result.sort_values('probability', ascending=True)
    print('final result : {}'. format(result))
    
          
          
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type = str, default = "flowers/test/17/image_03830.jpg", help = 'path to the flower file')
    parser.add_argument('--gpu', type=bool, default= True, help= 'training classifier')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'chosen model')
    parser.add_argument('--name_flower', type= str, default = 'cat_to_name.json', help = 'category label to category name mapping')
    parser.add_argument('--checkpoint_path', type = str, default = "./checkpoint.pth", help = 'location of saved model')
    return parser.parse_args()

          
          
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(nn.Linear(1024, 500),
                          nn.ReLU(),
                          nn.Dropout(p=0.5),
                          nn.Linear(500, 102),
                          nn.LogSoftmax(dim=1))
    model.classifier = classifier
                    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    return model
          

def process_image(images):    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(images)
    # Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height: 
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256, int(round(factor*256,0))))
     # Crop out the center 224x224 portion of the image.

    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
     # Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 0, 1))

    #tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return  np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = np.expand_dims(image, 0)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    model.eval()
    model, image_tensor = model.to(device), image_tensor.to(device)
    with torch.no_grad():
        output = model.forward(image_tensor)
    ps = torch.exp(output)
    probs, indices = ps.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices  = indices.cpu().numpy()[0]
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    
    return probs, classes

def flower_names(classes, class_to_idx, class_name):
    names = {}
    
    with open(class_name, 'r') as f:
        cat_to_name = json.load(f)
        
    for k in class_to_idx:
        names[class_to_idx[k]] = cat_to_name[k]
        
    return [names[c] for c in classes]

# Call to main function to run the program
if __name__ == '__main__':
    main()

