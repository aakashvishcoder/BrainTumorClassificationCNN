from xml.parsers.expat import model
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_ds = datasets.ImageFolder(root="C:\\Users\\aakas\\Downloads\\archive (1)\\Training",transform=transform)
testing_ds = datasets.ImageFolder(root="C:\\Users\\aakas\\Downloads\\archive (1)\\Testing",transform=transform)
batch_size = 32
training = DataLoader(training_ds,batch_size=batch_size, shuffle=True)
testing = DataLoader(testing_ds,batch_size=batch_size, shuffle=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   
        x = self.pool(F.relu(self.conv2(x)))    
        x = x.view(x.size(0), -1)               
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model,x_train,y_train,x_test,y_test):
    num_epochs = 100
    learning_rate = 0.001
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=learning_rate)
    min_loss = float('inf')
    best_weight = None
    for epoch in range(num_epochs):
        model.train()           
        running_loss = 0.0
        for data, target in training:
            data, target = data.to(device), target.to(device)
            optim.zero_grad()  
            output = model(data)  
            loss = loss_fn(output, target)  
            loss.backward()  
            optim.step()  
            running_loss += loss.item() * data.size(0)
        avg_loss = running_loss / len(training_ds)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for data, target in testing:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = total_loss / len(testing_ds)
            accuracy = 100 * correct / total
            
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_weight = model.state_dict()
    model.load_state_dict(best_weight)
    return f'Training complete. Best loss: {min_loss:.4f}, Accuracy: {accuracy:.2f}%'

train(NeuralNetwork(), training_ds, testing_ds, training, testing)
torch.save(NeuralNetwork().state_dict(), "model.pth")