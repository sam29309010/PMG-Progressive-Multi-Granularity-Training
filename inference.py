import os
import sys
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

net = torch.load("./bird_res101/model.pth")
# model = torch.load("./bird/model.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

transform_test = transforms.Compose([
    transforms.Scale((550, 550)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Homework Style Dataset


class BirdDataset(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = cv2.imread(img_path)
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        label = int(label.split('.')[0]) - 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Image path of the given homework datasets
DATA_PATH_ = r'../2021VRDL_HW1_datasets'
unlabeled_dataset = BirdDataset(
    annotations_file=os.path.join(DATA_PATH_, 'false_answer.txt'),
    img_dir=os.path.join(DATA_PATH_, 'testing_images/'),
    transform=transform_test,
    target_transform=None)
unlabeled_dataloader = torch.utils.data.DataLoader(
    unlabeled_dataset, batch_size=1, shuffle=False, num_workers=0)

# Predict Outputs
net.eval()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_list = list()
for batch_idx, (inputs, targets) in enumerate(unlabeled_dataloader):
    if (batch_idx % 100 == 0):
        print(batch_idx, "th images")
    if use_cuda:
        inputs, targets = inputs.to(device), targets.to(device)
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    output_1, output_2, output_3, output_concat = net(inputs)
    outputs_com = output_1 + output_2 + output_3 + output_concat
    outputs_com = outputs_com.cpu().detach().numpy().reshape(-1)
    output_list.append(outputs_com)
output_list = np.array(output_list)


# Generate Homework Submission
test_images = unlabeled_dataset.img_labels[0].values.tolist()
class_names = pd.read_csv(
    os.path.join(
        DATA_PATH_,
        'training_labels.txt'),
    sep=' ',
    header=None)[1].sort_values().unique()
myanswer_index = np.argmax(output_list, axis=1)
myanswer_class_names = class_names[myanswer_index]

submission = []
for image_name, image_pred in zip(test_images, myanswer_class_names):
    submission.append([image_name, image_pred])
np.savetxt('answer.txt', submission, fmt='%s')
