## Code 1 Sample


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load and preprocess CIFAR-10 dataset      
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset 
train_dataset = datasets.CIFAR10(root='/working', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='/working', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Parameters for Vision Transformer
img_size = 32
patch_size = 4
num_classes = 10
d_model = 64
num_heads = 4
num_layers = 4  # Reduced number of layers to simplify the model
mlp_dim = 128
dropout_rate = 0.1

# Function to create patches
def create_patches(images, patch_size):
    batch_size, channels, img_height, img_width = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size * patch_size)
    patches = patches.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channels * patch_size * patch_size)
    return patches

# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, num_layers, d_model, num_heads, mlp_dim, dropout_rate):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, d_model))
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        patches = create_patches(x, self.patch_size)
        embeddings = self.patch_embedding(patches)
        embeddings += self.position_embedding
        x = embeddings
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x.mean(dim=1)
        logits = self.mlp_head(x)
        return logits

# Instantiate and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(num_classes, num_layers, d_model, num_heads, mlp_dim, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate

# Training loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)

# Training and evaluating the model
num_epochs = 50  # Increased number of epochs for more training time
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

#####################################################################################################################################
#####################################################################################################################################