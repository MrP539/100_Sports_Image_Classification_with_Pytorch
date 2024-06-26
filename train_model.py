import sklearn.metrics
import torch.utils
import torch.utils.data
import torchvision
import torch
import os
import tqdm
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if torch.cuda.is_available(): gpu = True

loot_path = r"D:\machine_learning_AI_Builders\บท4\Classification\100_Sports_Image_Classification_with_Pytorch"

# เตรียม Data set
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((224,224),scale=(0.3,0.9)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.CenterCrop(150),
    torchvision.transforms.TrivialAugmentWide(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_path = os.path.join(loot_path,"data","train")
valid_path = os.path.join(loot_path,"data","valid")
test_path = os.path.join(loot_path,"data","test")

train_set = torchvision.datasets.ImageFolder(train_path,transform=train_transform)
valid_set = torchvision.datasets.ImageFolder(valid_path,transform=valid_transform)
test_set = torchvision.datasets.ImageFolder(test_path,transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=32,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False)

n_train = len(train_loader.dataset)
n_valid = len(valid_loader.dataset)
n_test = len(test_loader.dataset)

# เตรียม model

model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(in_features=2048,out_features=len(train_set.classes))

if gpu: model.cuda()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# pre csv logger

columns = ["epoch","train_loss","valid_loss","accuracy","f1_score"]
csv_df = pd.DataFrame(columns=columns)
csv_file_name = "result_log.csv" 
log_csv_path = os.path.join(loot_path,csv_file_name)
# # Train model

n_epochs = 50
bast_val_loss = float("inf")

for epoch in range(n_epochs):
    train_loss,valid_loss = 0,0
    valid_pred,valid_actual = [],[]

    model.train()
    for images_train,labels_train in tqdm.tqdm(train_loader):
        if gpu:
            image_train , label_train = images_train.cuda(),labels_train.cuda()
        optimizer.zero_grad()
        preds = model(image_train)
        loss = loss_function(preds,label_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * image_train.size(0)

    model.eval()
    for images_valid, labels_valid in tqdm.tqdm(valid_loader):
        if gpu:
            image_valid,label_valid = images_valid.cuda(),labels_valid.cuda()

        preds = model(image_valid)
        loss = loss_function(preds,label_valid)
        valid_loss += loss.item() * image_valid.size(0)

        valid_pred.extend(preds.argmax(dim = 1).cpu().numpy())
        valid_actual.extend(label_valid.cpu().numpy())

    # metics
    train_loss /= n_train
    valid_loss /= n_valid
    accuracy = sklearn.metrics.accuracy_score(y_pred=valid_pred,y_true=valid_actual)
    f1_score = sklearn.metrics.f1_score(y_pred=valid_pred,y_true=valid_actual,average="macro",zero_division=0)
    print(f"{epoch+1}/{n_epochs}\nTraing_loss : {train_loss}, Valid_loss : {valid_loss}, Accuracy : {accuracy}, F1-scroe : {f1_score}")

    #log_csv

    each_epoch_log = {f"{columns[0]}":int(epoch)+1,
                      f"{columns[1]}":train_loss,
                      f"{columns[2]}":valid_loss,
                      f"{columns[3]}":accuracy,
                      f"{columns[4]}":f1_score
                      }
    csv_df = pd.concat([csv_df,pd.DataFrame([each_epoch_log])],ignore_index=True,axis=0)
    csv_df.to_csv(log_csv_path,index=False)


    #save bast models

    if valid_loss < bast_val_loss:
        bast_val_loss = valid_loss
        model_path = os.path.join(loot_path,"model.pth")
        torch.save(model.state_dict(),model_path)
        print(f"***** Save Complete ******")

model_path = os.path.join(loot_path,"model.pth")
best_model = torchvision.models.resnet50(weights=None)
best_model.fc = torch.nn.Linear(in_features=2048,out_features=len(train_set.classes))
state_dict = torch.load(model_path)
best_model.load_state_dict(state_dict=state_dict)
if gpu: best_model.cuda()

best_model.eval()
y_pre,y_actual = [],[]
for img,label in tqdm.tqdm(valid_loader):
    if gpu:
        img,label = img.cuda(),label.cuda()
    with torch.no_grad():
        pred = best_model(img)
    y_pre.extend(pred.argmax(dim=1).cpu().numpy())
    y_actual.extend(label.cpu().numpy())

report = sklearn.metrics.classification_report(y_pred=y_pre,y_true=y_actual,zero_division=0)
accuracy_best_model_on_valid_set = sklearn.metrics.accuracy_score(y_pred=y_pre,y_true=y_actual)

print(report)
print(f"Accuracy best model : {accuracy_best_model_on_valid_set}")

data_log = pd.read_csv(log_csv_path)

plt.figure(figsize=(6,6))
plt.plot(data_log.epoch,data_log.train_loss,label= "train_loss",color = "blue",marker ="x")
plt.plot(data_log.epoch,data_log.valid_loss,label= "valid_loss",color = "red",marker ="x")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss-epoch")
plt.show()

plt.figure(figsize=(6,6))
plt.plot(data_log.epoch,data_log.accuracy,label= "accuracy",color = "blue",marker ="x")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("accuracy")
plt.show()

plt.figure(figsize=(6,6))
plt.plot(data_log.epoch,data_log.f1_score,label= "f1_score",color = "blue",marker ="x")
plt.xlabel("epoch")
plt.ylabel("f1_score")
plt.title("f1_score")
plt.show()
