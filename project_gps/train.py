import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BeamDataset
from model import BeamMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = BeamDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BeamMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50
print(f"开始在 {device} 上训练...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / len(dataset)
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} Acc: {acc:.2f}%")

# 5. 保存模型
torch.save(model.state_dict(), "beam_model.pth")
print("训练完成，模型已保存为 beam_model.pth")