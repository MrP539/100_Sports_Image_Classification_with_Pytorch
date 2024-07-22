import torch
import torchvision
from PIL import Image
import os
import numpy as np
import cv2

if torch.cuda.is_available():
    gpu = True

img_path_cv2 = "test.png"

test_set_path = os.path.join("data","test")
test_set = torchvision.datasets.ImageFolder(test_set_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path_cv2)

img_tensor = transform(img)
img_tensor.unsqueeze_(0)

if gpu:
    img_tensor = img_tensor.cuda()


model = torchvision.models.resnet50(weights = None)
model.fc = torch.nn.Linear(in_features=2048,out_features=len(test_set.classes))

model_path = os.path.join("model","pretrain_resnet50_optimazer(SGD)","model.pth")
state_dic = torch.load(model_path)
model.load_state_dict(state_dict=state_dic)

if gpu:
    model = model.cuda()

with torch.no_grad():
    pred = model(img_tensor)

class_index = pred.argmax(dim = 1).cpu().numpy().item()
class_text = test_set.classes[class_index]

img_cv2 = cv2.imread(img_path_cv2)
img_cv2_resize = cv2.resize(img_cv2,(300,300))
cv2.putText(img_cv2_resize,f"{class_text}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
cv2.imshow('result',img_cv2_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
