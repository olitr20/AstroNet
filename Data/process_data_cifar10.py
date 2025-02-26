import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import *
import numpy as np


normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
transf_cifar = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=False, transform=transf_cifar)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=False, transform=transf_cifar)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

lst_date_0, lst_date_1, lst_date_2, lst_date_3, lst_date_4, lst_date_5, lst_date_6, lst_date_7, lst_date_8, lst_date_9 = [], [], [], [], [], [], [], [], [], []
lst_lable_0, lst_lable_1, lst_lable_2, lst_lable_3, lst_lable_4, lst_lable_5, lst_lable_6, lst_lable_7, lst_lable_8, lst_lable_9   = [], [], [], [], [], [], [], [], [], []

for batch_id, (image, label) in enumerate(trainloader):
    print(batch_id)
    if label == 0:
        image = image.view(3, 32, 32)
        lst_date_0.append(np.array(image))
        lst_lable_0.append(0)
    elif label == 1:
        image = image.view(3, 32, 32)
        lst_date_1.append(np.array(image))
        lst_lable_1.append(1)
    elif label == 2:
        image = image.view(3, 32, 32)
        lst_date_2.append(np.array(image))
        lst_lable_2.append(2)
    elif label == 3:
        image = image.view(3, 32, 32)
        lst_date_3.append(np.array(image))
        lst_lable_3.append(3)
    elif label == 4:
        image = image.view(3, 32, 32)
        lst_date_4.append(np.array(image))
        lst_lable_4.append(4)
    elif label == 5:
        image = image.view(3, 32, 32)
        lst_date_5.append(np.array(image))
        lst_lable_5.append(5)
    elif label == 6:
        image = image.view(3, 32, 32)
        lst_date_6.append(np.array(image))
        lst_lable_6.append(6)
    elif label == 7:
        image = image.view(3, 32, 32)
        lst_date_7.append(np.array(image))
        lst_lable_7.append(7)
    elif label == 8:
        image = image.view(3, 32, 32)
        lst_date_8.append(np.array(image))
        lst_lable_8.append(8)
    elif label == 9:
        image = image.view(3, 32, 32)
        lst_date_9.append(np.array(image))
        lst_lable_9.append(9)

train_data0 = np.array(lst_date_0)
train_data0 = torch.from_numpy(train_data0)
train_label0 = np.array(lst_lable_0)
train_label0 = torch.from_numpy(train_label0)
Train_Dataset0 = TensorDataset(train_data0, train_label0)
torch.save(Train_Dataset0, './Train0''.t7')

train_data1 = np.array(lst_date_1)
train_data1 = torch.from_numpy(train_data1)
train_label1 = np.array(lst_lable_1)
train_label1 = torch.from_numpy(train_label1)
Train_Dataset1 = TensorDataset(train_data1, train_label1)
torch.save(Train_Dataset1, './Train1''.t7')

train_data2 = np.array(lst_date_2)
train_data2 = torch.from_numpy(train_data2)
train_label2 = np.array(lst_lable_2)
train_label2 = torch.from_numpy(train_label2)
Train_Dataset2 = TensorDataset(train_data2, train_label2)
torch.save(Train_Dataset2, './Train2''.t7')

train_data3 = np.array(lst_date_3)
train_data3 = torch.from_numpy(train_data3)
train_label3 = np.array(lst_lable_3)
train_label3 = torch.from_numpy(train_label3)
Train_Dataset3 = TensorDataset(train_data3, train_label3)
torch.save(Train_Dataset3, './Train3''.t7')

train_data4 = np.array(lst_date_4)
train_data4 = torch.from_numpy(train_data4)
train_label4 = np.array(lst_lable_4)
train_label4 = torch.from_numpy(train_label4)
Train_Dataset4 = TensorDataset(train_data4, train_label4)
torch.save(Train_Dataset4, './Train4''.t7')

train_data5 = np.array(lst_date_5)
train_data5 = torch.from_numpy(train_data5)
train_label5 = np.array(lst_lable_5)
train_label5 = torch.from_numpy(train_label5)
Train_Dataset5 = TensorDataset(train_data5, train_label5)
torch.save(Train_Dataset5, './Train5''.t7')

train_data6 = np.array(lst_date_6)
train_data6 = torch.from_numpy(train_data6)
train_label6 = np.array(lst_lable_6)
train_label6 = torch.from_numpy(train_label6)
Train_Dataset6 = TensorDataset(train_data6, train_label6)
torch.save(Train_Dataset6, './Train6''.t7')

train_data7 = np.array(lst_date_7)
train_data7 = torch.from_numpy(train_data7)
train_label7 = np.array(lst_lable_7)
train_label7 = torch.from_numpy(train_label7)
Train_Dataset7 = TensorDataset(train_data7, train_label7)
torch.save(Train_Dataset7, './Train7''.t7')

train_data8 = np.array(lst_date_8)
train_data8 = torch.from_numpy(train_data8)
train_label8 = np.array(lst_lable_8)
train_label8 = torch.from_numpy(train_label8)
Train_Dataset8 = TensorDataset(train_data8, train_label8)
torch.save(Train_Dataset8, './Train8''.t7')

train_data9 = np.array(lst_date_9)
train_data9 = torch.from_numpy(train_data9)
train_label9 = np.array(lst_lable_9)
train_label9 = torch.from_numpy(train_label9)
Train_Dataset9 = TensorDataset(train_data9, train_label9)
torch.save(Train_Dataset9, './Train9''.t7')


trainloader0 = DataLoader(torch.load('./Train0.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader1 = DataLoader(torch.load('./Train1.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader2 = DataLoader(torch.load('./Train2.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader3 = DataLoader(torch.load('./Train3.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader4 = DataLoader(torch.load('./Train4.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader5 = DataLoader(torch.load('./Train5.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader6 = DataLoader(torch.load('./Train6.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader7 = DataLoader(torch.load('./Train7.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader8 = DataLoader(torch.load('./Train8.t7', weights_only=False), batch_size=1, shuffle=False)
trainloader9 = DataLoader(torch.load('./Train9.t7', weights_only=False), batch_size=1, shuffle=False)

train, val = [], []
train_lable, val_lable = [], []
for batch_id, (image, label) in enumerate(trainloader0):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader1):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader2):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader3):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader4):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader5):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader6):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader7):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader8):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

for batch_id, (image, label) in enumerate(trainloader9):
    if batch_id < 800:
        image = image.view(3, 32, 32)
        train.append(np.array(image))
        train_lable.append(label)
    elif 800 <= batch_id < 1000:
        image = image.view(3, 32, 32)
        val.append(np.array(image))
        val_lable.append(label)

train_data = np.array(train)
train_data = torch.from_numpy(train_data)
train_label = np.array(train_lable)
train_label = torch.from_numpy(train_label)
Train_Dataset = TensorDataset(train_data, train_label)
torch.save(Train_Dataset, './Train_CIFAR10''.t7')
val_data = np.array(val)
val_data = torch.from_numpy(val_data)
val_label = np.array(val_lable)
val_label = torch.from_numpy(val_label)
Train_Dataset = TensorDataset(val_data, val_label)
torch.save(Train_Dataset, './Val_CIFAR10''.t7')

trainloader = torch.utils.data.DataLoader(torch.load('./Train_CIFAR10.t7', weights_only=False), batch_size=1, shuffle=False)
valloader = torch.utils.data.DataLoader(torch.load('./Val_CIFAR10.t7', weights_only=False), batch_size=1, shuffle=False)
for batch_id, (image, target) in enumerate(trainloader):
    print(batch_id)