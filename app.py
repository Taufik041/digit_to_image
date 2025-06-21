# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
import random

# Define the CNN model
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load model
@st.cache_resource
def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load("saved_model/mnist_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Load both training and test MNIST datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Combine both sets
full_dataset = ConcatDataset([train_set, test_set])

# UI
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
generate = st.button("Generate Images")

if generate:
    # Filter all images matching selected digit
    filtered = [full_dataset[i][0] for i in range(len(full_dataset)) if full_dataset[i][1] == digit]
    if len(filtered) < 5:
        st.warning("Not enough samples available for this digit.")
    else:
        samples = random.sample(filtered, 5)

        st.subheader(f"Generated images of digit {digit}")
        cols = st.columns(5)
        for idx, (img, col) in enumerate(zip(samples, cols)):
            img = img.squeeze().numpy()
            col.image(img, caption=f"Sample {idx+1}", use_container_width=True, clamp=True)
