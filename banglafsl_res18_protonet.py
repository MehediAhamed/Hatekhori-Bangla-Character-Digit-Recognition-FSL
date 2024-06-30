from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torchvision
import torch.utils.data

random_seed = 30
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_way = 5
n_shot = 1
n_query = 10

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
DEVICE = "cuda"
n_workers = 0

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class BengaliCharactersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    self.samples.append((os.path.join(class_dir, img_file), class_name))

        # Debug: Print the first few samples
        print("First few samples:")
        print(self.samples[:5])  # Adjust the number to print more or fewer samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # Convert label from string to integer
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]


image_size = 84  # Adjusted for ResNet, which typically expects larger input sizes

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_set = BengaliCharactersDataset(root_dir='content/data/BanglaLekha_Isolated_mod/Train', transform=train_transforms)
test_set = BengaliCharactersDataset(root_dir='content/data/BanglaLekha_Isolated_mod/Test', transform=test_transforms)

from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader


n_tasks_per_epoch = 500
n_validation_tasks = 100


train_set_size = int(len(train_set) * 0.8)
val_set_size = len(train_set) - train_set_size
train_set, val_set = torch.utils.data.random_split(train_set, [train_set_size, val_set_size])

train_set.get_labels = lambda: [
    instance[1] for instance in train_set
]

val_set.get_labels = lambda: [
    instance[1] for instance in val_set
]

# Those are special batch samplers that sample few-shot classification tasks with a pre-defined shape
train_sampler = TaskSampler(
    train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
)
val_sampler = TaskSampler(
    val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)

print('sampling done')

# Finally, the DataLoader. We customize the collate_fn so that batches are delivered
# in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

print(train_set_size,val_set_size)

# from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
# from easyfsl.modules import resnet12


# convolutional_network = resnet12()
# few_shot_classifier = PrototypicalNetworks(convolutional_network).to(DEVICE)

import torchvision.models as models
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
import torch.nn as nn
import torch

# Load a pretrained ResNet18 model
pretrained_resnet18 = models.resnet18(pretrained=True)

# Adjust the final layer for 50 Bengali characters
num_classes = 50  # Since you have 50 distinct characters
pretrained_resnet18.fc = nn.Linear(pretrained_resnet18.fc.in_features, num_classes)

# Ensure the model is set to the correct device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE= "cpu"
DEVICE = torch.device("cuda")
pretrained_resnet18 = pretrained_resnet18.to(DEVICE)

# Create the few-shot classifier with the modified ResNet18
# Note: Ensure that PrototypicalNetworks or any other few-shot method you choose
# can work effectively with the character recognition task.
few_shot_classifier = PrototypicalNetworks(pretrained_resnet18).to(DEVICE)

print('model created')
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


LOSS_FUNCTION = nn.CrossEntropyLoss()

n_epochs = 40
scheduler_milestones = [120, 160]
scheduler_gamma = 0.1
learning_rate = 1e-2
tb_logs_dir = Path(".")

train_optimizer = SGD(
    few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
)
train_scheduler = MultiStepLR(
    train_optimizer,
    milestones=scheduler_milestones,
    gamma=scheduler_gamma,
)

tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

print('scheduler created')

def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer
):
    all_loss = []
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            classification_scores = model(query_images.to(DEVICE))

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))

            loss.backward()

            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)
print('epoch created')

PATH = 'content/1shot_5way_BanglaPrototypicalNetworks_BanglaLekha_Isolated.pth'
# few_shot_classifier.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
few_shot_classifier.load_state_dict(torch.load(PATH))

print('weights loaded')













def predict_image_with_support_set(query_image_path):
    # Load and transform the query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_image_tensor = image_transforms(query_image).unsqueeze(0)  # Add batch dimension

    # Set the model in evaluation mode
    few_shot_classifier.eval()

    # Use the model to predict
    with torch.no_grad():
        few_shot_classifier.process_support_set(support_images_tensor.to(DEVICE), support_labels_tensor.to(DEVICE))
        outputs = few_shot_classifier(query_image_tensor.to(DEVICE))
        print('tensor done')
        

    probabilities = torch.softmax(outputs, dim=1)
    predicted_label_idx = probabilities.argmax(1).item()
    predicted_label = support_labels[predicted_label_idx] + 1  # Adjust label to match original range

    return predicted_label



import torch
from torchvision import transforms
from PIL import Image
import os


label_to_bengali = {
    1: "অ",
    2: "আ",
    3: "ই",
    4: "ঈ",
    5: "উ",
    6: "ঊ",
    7: "ঋ",
    8: "এ",
    9: "ঐ",
    10: "ও",
    11: "ঔ",
    12: "ক",
    13: "খ",
    14: "গ",
    15: "ঘ",
    16: "ঙ",
    17: "চ",
    18: "ছ",
    19: "জ",
    20: "ঝ",
    21: "ঞ",
    22: "ট",
    23: "ঠ",
    24: "ড",
    25: "ঢ",
    26: "ণ",
    27: "ত",
    28: "থ",
    29: "দ",
    30: "ধ",
    31: "ন",
    32: "প",
    33: "ফ",
    34: "ব",
    35: "ভ",
    36: "ম",
    37: "য",
    38: "র",
    39: "ল",
    40: "শ",
    41: "ষ",
    42: "স",
    43: "হ",
    44: "ড়",
    45: "ঢ়",
    46: "য়",
    47: "ৎ",
    48: "ং",
    49: "ঃ",
    50: "ঁ",
    51: "০",
    52: "১",
    53: "২",
    54: "৩",
    55: "৪",
    56: "৫",
    57: "৬",
    58: "৭",
    59: "৮",
    60: "৯",
}


# Function to print the Bengali character for a given label
def print_bengali_character(label):
    character = label_to_bengali.get(label, "Unknown label")
    print(character)


# Define the image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((84, 84)),  # Resize the image to what the model expects
    transforms.ToTensor(),        # Convert the image to a tensor
])

# Define the root directory of your support images
root_dir = 'content/data/BanglaLekha_Isolated_mod/Test/'

# Generate support set
support_images = []
support_labels = []

# Assuming each class directory contains images for that class
for label in range(1, 61):  # Labels from 1 to 60
    class_dir = os.path.join(root_dir, str(label))
    if not os.path.exists(class_dir):
        continue  # Skip if the class directory doesn't exist

    # List all images in the class directory
    img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    for img_file in img_files[:1]:  # Take only the first image for simplicity; adjust as needed
        img_path = os.path.join(class_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        support_images.append(image_transforms(image))
        support_labels.append(label - 1)  # Adjust label to start from 0

# Convert lists to tensors
support_images_tensor = torch.stack(support_images)
support_labels_tensor = torch.tensor(support_labels)




query_image_path = 'content/drawing.jpg'
predicted_label = predict_image_with_support_set(query_image_path)
print("Predicted label: ")
print_bengali_character(predicted_label)
