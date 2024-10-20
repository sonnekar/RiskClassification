import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_pickle('df.pkl')
df['report'] = df['PNT_NM'] + df['QUALIFIER_TXT'] + df['PNT_ATRISKNOTES_TX'] + df['PNT_ATRISKFOLWUPNTS_TX']
reports = df['report'].values
danger_magnitudes = df['ovr_danger'].values

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name)

def get_embeddings(reports):
    inputs = tokenizer(reports.tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = transformer_model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
    return embeddings.numpy()

embeddings = get_embeddings(reports)

X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(danger_magnitudes, dtype=torch.float32)  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class DangerMagnitudeNN(nn.Module):
    def __init__(self):
        super(DangerMagnitudeNN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 128)      # Input layer
        self.fc_proj = nn.Linear(X.shape[1], 128)  # Projection layer to match dimensions
        self.bn1 = nn.BatchNorm1d(128)             # Batch Normalization after input
        self.fc2 = nn.Linear(128, 64)              # Hidden layer
        self.bn2 = nn.BatchNorm1d(64)              # Batch Normalization after hidden layer
        self.fc3 = nn.Linear(64, 32)               # Additional hidden layer
        self.fc4 = nn.Linear(32, 1)                # Output layer (for regression)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)             # Dropout layer for regularization

    def forward(self, x):
        # Project input to match fc1 dimensions for residual connection
        residual = self.fc_proj(x)  # Projection layer to match dimensions for residual
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x += residual  # Residual connection (skip connection)

        # Hidden Layer 1 -> BatchNorm -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Hidden Layer 2 -> ReLU -> Output
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

model = DangerMagnitudeNN()

criterion = nn.MSELoss()  # Use Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    # Validation printing
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs.squeeze(), batch_y).item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

torch.save(model.state_dict(), 'danger_magnitude_model.pth')

# Evaluate the model on validation data
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val).squeeze()

mae = mean_absolute_error(y_val.numpy(), y_val_pred.numpy())
print(f'MAE score on the validation set: {mae:.4f}\nVariation in set = {np.array(y_val).std()}')

plt.plot(y_val.numpy()[::5])
plt.plot(y_val_pred.numpy()[::5])
plt.show()