import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torchvision
from exceptions import NoSuchNameError , NoIndexError
from modelNet import Net

def load_model(model_name, Net: object):
    
    try:
        if '.pt' in model_name: #for saved model (.pt)
            state_dict = torch.load(model_name)
            # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            # if torch.typename(m) == 'OrderedDict':

            #     """
            #     if you want to use customized model that has a type 'OrderedDict',
            #     you shoud load model object as follows:
                
            #     from Net import Net()
            #     model=Net()
            #     """
            #     print('SDNBDHIAB EFUIEBFUEIAFUEOQFEHFUOEFQ')
            model = Net()
            model.load_state_dict(state_dict)
            # else:
            #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            #     model = m

        elif hasattr(models, model_name): #for pretrained model (ImageNet)
            model = getattr(models, model_name)(pretrained=True)

        # model.eval()
        # if cuda_available():
        #     model.cuda()
    except:
        raise ValueError(f'Not unvalid model was loaded: {model_name}')
        
    return model

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_image(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255

    return img

def preprocess_image(img):
    mean = 0.5  # pode ajustar dependendo do treino
    std = 0.5

    # converte para grayscale fazendo média dos canais
    gray = np.mean(img, axis=2)  # [H, W]
    gray = (gray - mean) / std
    gray = np.expand_dims(gray, axis=0)  # [1, H, W]

    tensor = torch.from_numpy(gray).unsqueeze(0).float()

    return Variable(tensor, requires_grad=False)

def save(mask, img, img_path, model_path):

    mask = (mask - np.min(mask)) / np.max(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    gradcam = 1.0 * heatmap + img
    gradcam = gradcam / np.max(gradcam)

    index = img_path.find('/')
    index2 = img_path.find('.')

    if '.pt' in model_path:
        dotindex = model_path.find('.')
        model_path = model_path[:dotindex]
        
    path = 'result/' + img_path[index + 1:index2] +'/'+model_path
    if not (os.path.isdir(path)):
        os.makedirs(path)

    gradcam_path = path + "/gradcam.png"
    n = 1
    while True:

        if os.path.exists(path + "/gradcam.png"):
            gradcam_path = path + "/gradcam" + str(n) + ".png" 
            n += 1
            continue

        cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))
        break

    
def is_int(v):
    v = str(v).strip()
    return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()

def _exclude_layer(layer):

    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False

def choose_tlayer(model):
    name_to_num = {}
    num_to_layer = {}
    for idx, data in enumerate(model.named_modules()):        
        name, layer = data
        if _exclude_layer(layer):
            continue
        
        name_to_num[name] = idx
        num_to_layer[idx] = layer
        print(f'[ Number: {idx},  Name: {name} ] -> Layer: {layer}\n')
   
    print('\n<<-------------------------------------------------------------------->>')
    print('\n<<      You sholud not select [classifier module], [fc layer] !!      >>')
    print('\n<<-------------------------------------------------------------------->>\n')

    a = input(f'Choose "Number" or "Name" of a target layer: ')

    
    if a.isnumeric() == False:
        a = name_to_num[a]
    else:
        a = int(a)
    try:
        t_layer = num_to_layer[a]
        return t_layer    
    except IndexError:
        raise NoIndexError('Selected index (number) is not allowed.')
    except KeyError:
        raise NoSuchNameError('Selected name is not allowed.')
