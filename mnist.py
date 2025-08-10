import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# transform: convert PIL images to tensors and normalize to [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean, std for MNIST
])

# download/train dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# data loaders: iterate in batches, shuffle training data
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False
)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # MNIST images are 28×28 = 784 inputs
        self.fc1 = nn.Linear(28*28, 128)  # hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)     # 10 output classes

    def forward(self, x):
        # x: [batch_size, 1, 28, 28] → flatten to [batch_size, 784]
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()          # combines LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 3

for epoch in range(1, num_epochs + 1):
    model.train()   # set to training mode (affects dropout, batchnorm, etc.)
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and optimization step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/100:.4f}")
            running_loss = 0.0


