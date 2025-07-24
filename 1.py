import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 images
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(
    root='/home/kami/Documents/datasets/',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  # C1: 28x28 output
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2: 14x14 output
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # C3: 10x10 output
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4: 5x5 output
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # C5
        self.fc2 = nn.Linear(120, 84)  # F6
        self.fc3 = nn.Linear(84, 10)  # Output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

st = time.time()
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

et = time.time()
print("Time Elapsed:" , et - st)
print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), '/home/kami/Documents/datasets/mnist_lenet5.pth')
