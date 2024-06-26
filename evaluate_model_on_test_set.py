import sklearn.metrics
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import os
import torch.utils
import tqdm
import sklearn
import numpy as np
import seaborn as sns
from create_confusion_matrix import *
import pandas as pd

gpu = torch.cuda.is_available()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_set_path = os.path.join("data","test")
test_set = torchvision.datasets.ImageFolder(test_set_path,transform=transform)

test_loader = torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False,num_workers=0)

model = torchvision.models.resnet50(weights = None)
model.fc = torch.nn.Linear(in_features=2048,out_features=len(test_set.classes))
model_path = os.path.join(r"D:\machine_learning_AI_Builders\บท4\Classification\100_Sports_Image_Classification_with_Pytorch\model\pretrain_resnet50_optimazer(SGD)","model.pth")
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)    

if gpu:
    model.cuda()

y_pred,y_actual = [],[]
model.eval()
for images_test, labels_test in tqdm.tqdm(test_loader):
    if gpu:
        image_test, label_test = images_test.cuda(),labels_test.cuda()
    with torch.no_grad():
        pred = model(image_test)
    y_pred.extend(pred.argmax(dim=1).cpu().numpy())
    y_actual.extend(label_test.cpu().numpy())

accuracy = sklearn.metrics.accuracy_score(y_pred=y_pred,y_true=y_actual)
f1_score = sklearn.metrics.f1_score(y_pred=y_pred,y_true=y_actual,average="macro",zero_division=0)
report = sklearn.metrics.classification_report(y_pred=y_pred,y_true=y_actual,zero_division=0)
df = pd.DataFrame([report])
df.to_csv("evaluate_on_test_set.csv")
print(f"{report} \n Accuracy: {accuracy}, F1_score {f1_score}")

CREATE_CONFUSION_MATRICS(y_actual=y_actual,y_pred=y_pred)

