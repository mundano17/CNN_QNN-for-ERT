import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
import pennylane as qml

# Load your CSV
data = pd.read_csv("synthetic_ert_multi_anomalies.csv")

# Feature engineering for ERT data
# Add log transformation for resistivity values (common in geophysics)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Feature enrichment - create polynomial features for spatial relationships
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Better normalization for geophysical data
scaler_X = PowerTransformer(method='yeo-johnson')
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Log-transform target before standardization (common for resistivity data)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

y_mean = y_train_log.mean()
y_std = y_train_log.std()
y_train_scaled = (y_train_log - y_mean) / y_std
y_test_scaled = (y_test_log - y_mean) / y_std

# Convert to Torch Tensors with proper shape
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Define a quantum device
n_qubits = 4  # Number of qubits to use
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum circuit for feature extraction
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode the classical data into quantum states
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Apply parameterized quantum gates
    for i in range(n_qubits):
        qml.RZ(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)

    # Entangle qubits
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])

    # Second layer of parameterized gates
    for i in range(n_qubits):
        qml.RZ(weights[i+n_qubits, 0], wires=i)
        qml.RY(weights[i+n_qubits, 1], wires=i)

    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Quantum layer as a PyTorch module
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_features):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_features = min(n_features, n_qubits)  # Limit to number of qubits

        # Initialize quantum weights
        weight_shapes = {"weights": (2*n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # If input features > qubits, add a linear layer to reduce dimensions
        self.feature_map = nn.Linear(n_features, n_qubits) if n_features > n_qubits else None

    def forward(self, x):
        batch_size = x.shape[0]

        # If needed, map features to match qubit count
        if self.feature_map:
            x = self.feature_map(x)
        else:
            # Only use as many features as we have qubits
            x = x[:, :self.n_qubits]

        # Process each sample in the batch
        q_out = torch.zeros(batch_size, self.n_qubits, device=x.device)
        for i in range(batch_size):
            q_out[i] = self.qlayer(x[i])

        return q_out

# Hybrid Quantum-Classical model
class HybridERTQuantumCNN(nn.Module):
    def __init__(self, input_dim):
        super(HybridERTQuantumCNN, self).__init__()

        self.input_dim = input_dim

        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU()  # SiLU/Swish activation instead of LeakyReLU
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.Sigmoid()
        )

        # Deep feature extraction with dilated convolutions (better for capturing spatial patterns)
        self.dilation_block = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.SiLU()
        )

        # Residual block 1
        self.res_block1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128)
            ),
            nn.SiLU()
        ])

        # Feature reduction and global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Quantum processing layer
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_features=128)

        # Final prediction layers with skip connection
        self.fc1 = nn.Linear(n_qubits, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Initial convolution
        out = self.conv1(x)

        # Apply spatial attention
        attention = self.spatial_attention(out)
        out = out * attention  # Element-wise multiplication

        # Dilated convolutions for spatial context
        out = self.dilation_block(out)

        # Residual block
        identity = out
        out = self.res_block1[0](out)
        out = out + identity
        out = self.res_block1[1](out)

        # Global context
        out = self.global_context(out)

        # Quantum processing
        quantum_out = self.quantum_layer(out)

        # Final prediction with skip connections
        x1 = self.fc1(quantum_out)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)

        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)

        x3 = self.fc3(x2)

        return x3

# Create a hybrid ensemble
class HybridERTEnsemble(nn.Module):
    def __init__(self, input_dim, n_models=2):
        super(HybridERTEnsemble, self).__init__()
        self.models = nn.ModuleList([
            HybridERTQuantumCNN(input_dim) for _ in range(n_models)
        ])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # Average the predictions
        return torch.mean(torch.stack(outputs), dim=0)

# Create model, loss function, and optimizer
input_dim = X_train.shape[1]
model = HybridERTEnsemble(input_dim, n_models=2)  # Reduced to 2 models due to quantum overhead
criterion = nn.HuberLoss(delta=0.3)  # More robust to outliers than MSE

# Mixed precision for faster training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Optimizer with weight decay and gradient clipping
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

# One-cycle learning rate policy - helps converge faster and generalize better
steps_per_epoch = 1
epochs = 200  # Reduced epochs due to slower quantum simulation
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.0005,  # Lower max learning rate
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,  # Spend 30% of training time warming up
    div_factor=25,   # Initial LR = max_lr/25
    final_div_factor=1000  # Final LR = max_lr/1000
)

# Training with enhanced monitoring and early stopping
best_loss = float('inf')
patience = 30
patience_counter = 0
history = {'train_loss': [], 'val_loss': []}

for epoch in range(epochs):
    # Training phase
    model.train()

    # Mixed precision training step
    if scaler:
        with torch.cuda.amp.autocast():
            outputs = model(X_train_tensor).squeeze()
            train_loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(X_train_tensor).squeeze()
        train_loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor).squeeze()
        val_loss = criterion(val_outputs, y_test_tensor)

    # Record history
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss.item())

    # Early stopping with model checkpointing
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_hybrid_ert_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {train_loss.item():.6f} '
              f'Val Loss: {val_loss.item():.6f} '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

# Load best model for evaluation
model.load_state_dict(torch.load('best_hybrid_ert_model.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

# Reverse normalization and log transform to get original scale
predictions_unscaled = (predictions * y_std) + y_mean
y_test_unscaled = (y_test_np * y_std) + y_mean

# Reverse log transform
predictions_final = np.expm1(predictions_unscaled)
y_test_final = np.expm1(y_test_unscaled)

# Plot learning curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_test_final, predictions_final, alpha=0.7)
plt.plot([min(y_test_final), max(y_test_final)],
         [min(y_test_final), max(y_test_final)], 'r--')
plt.xlabel('Actual Resistivity')
plt.ylabel('Predicted Resistivity')
plt.title('Predicted vs Actual Resistivity')
plt.grid(True)

# Plot residuals to check for bias
plt.subplot(1, 3, 3)
residuals = predictions_final - y_test_final
plt.scatter(y_test_final, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Resistivity')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print metrics
mse = np.mean((predictions_final - y_test_final) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions_final - y_test_final))
r2 = 1 - (np.sum((y_test_final - predictions_final) ** 2) /
          np.sum((y_test_final - np.mean(y_test_final)) ** 2))

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
