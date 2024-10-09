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


model = torch.load('wine_quality.pth')
model.eval()  # Set the model to evaluation mode



sample_data = {
    'fixed acidity': [7.4],
    'volatile acidity': [0.7],
    'citric acid': [0.0],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
}
sample_df = pd.DataFrame(sample_data)

\
scaler = StandardScaler()
#
scaler.fit(sample_df)


scaled_input = scaler.transform(sample_df)

input_tensor = torch.FloatTensor(scaled_input)


with torch.no_grad():
    prediction = model(input_tensor)

print(f"Predicted wine quality: {prediction.item():.2f}")

