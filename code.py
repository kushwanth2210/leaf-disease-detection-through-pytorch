# just a test

'''
this is a model which sepaerates training data and testing data and graphs are in this model 
'''
!pip install git+https://github.com/zer0sh0t/zer0t0rch
'''
import libraries 
'''
import torch
from torch import nn
import matplotlib.pyplot as plt
from zer0t0rch import Zer0t0rchWrapper
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from zer0t0rch.utils import clear_cache, get_accuracy
from PIL import Image

clear_cache()
'''
transforming numpy data to tensors and resizing the image
'''
T = transforms.Compose(
    [
     transforms.Resize((224, 224)),
     transforms.ToTensor()
    ]
)
'''
loading the datasets and seperating images and labels and listing them 
'''

data = ImageFolder('plantvillage dataset', transform=T)
class_list = data.classes
print(len(data), len(class_list), class_list[:10])
img, label = data[50]
print(class_list[label])
plt.imshow(img.permute(1, 2, 0))
'''
this is a pretrained model names resnet18 our data is passing into this model 
'''
model = models.resnet18(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 38)
'''
Caluculating the loss with validation percentage 20%
'''

loss_fn = nn.CrossEntropyLoss()
metric_fns = {'acc': get_accuracy}
our_model = Zer0t0rchWrapper(model)
our_model.prepare_data(data, batch_size=64, val_pct=0.2)
our_model.compile(loss_fn, metric_fns)
our_model.fit(num_epochs=2, plot_graphs=True)
'''
predicting the image and the result we get is probability by giving the image path 

'''
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = T(img).to(our_model.device)
    preds = our_model.predict(img.unsqueeze(0))  
    prob, label = preds.softmax(-1).max(-1)
    print(f'probability: {prob.item()} \n Accuracy:{prob.item()*100} \n{class_list[label.item()]}')