import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

class WineQualityNN(nn.Module):
    def __init__(self, input_size):
        super(WineQualityNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = torch.load('wine_quality.pth', weights_only=False)
model.eval() 


sample_df = pd.read_csv('sample_wine_data.csv')


scaler = StandardScaler()
scaled_input = scaler.fit_transform(sample_df)

input_tensor = torch.FloatTensor(scaled_input)

with torch.no_grad():
    prediction = model(input_tensor)

print(f"Predicted wine quality: {prediction.item():.2f}")
