import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 1. Define the Model
class DrowsinessClassifier(nn.Module):
    def __init__(self):
        super(DrowsinessClassifier, self).__init__()
        # Input: [EAR, PERCLOS, Pitch, MAR]
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output: [Safe, Danger]

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # Logits

# 2. Generate Synthetic Data
def generate_data(n_samples=2000):
    X = []
    y = []
    
    for _ in range(n_samples):
        # Randomly decide if this sample should be Safe (0) or Danger (1)
        label = np.random.randint(0, 2)
        
        if label == 0:  # SAFE
            ear = np.random.uniform(0.25, 0.45)
            perclos = np.random.uniform(0.0, 0.15)
            # head_down signal: 0 when normal
            pitch = np.random.uniform(0.0, 2.0)
            mar = np.random.uniform(0.0, 0.5)
        else:  # DANGER (one or more indicators)
            condition = np.random.randint(0, 3)
            if condition == 0: # Sleepy eyes
                ear = np.random.uniform(0.05, 0.22)
                perclos = np.random.uniform(0.25, 0.8)
                pitch = np.random.uniform(0.0, 2.0)
                mar = np.random.uniform(0.0, 0.4)
            elif condition == 1: # Nodding
                ear = np.random.uniform(0.2, 0.35)
                perclos = np.random.uniform(0.1, 0.3)
                # head_down deviation becomes positive when nodding
                pitch = np.random.uniform(8.0, 30.0)
                mar = np.random.uniform(0.0, 0.4)
            else: # Yawning
                ear = np.random.uniform(0.15, 0.3)
                perclos = np.random.uniform(0.1, 0.4)
                pitch = np.random.uniform(0.0, 2.0)
                mar = np.random.uniform(0.6, 1.0)
                
        X.append([ear, perclos, pitch, mar])
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("Generating synthetic data...")
    X_train, y_train = generate_data()
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    print("Training model...")
    model = DrowsinessClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Test on a few manual cases
    model.eval()
    test_cases = [
        [0.35, 0.05, 0.0, 0.1],  # Safe
        [0.15, 0.40, 5.0, 0.2],  # Drowsy (eyes)
        [0.30, 0.10, 25.0, 0.1], # Drowsy (head)
        [0.30, 0.10, 0.0, 0.8],  # Drowsy (yawn)
    ]
    print("\nSanity Check:")
    with torch.no_grad():
        for tc in test_cases:
            logits = model(torch.tensor([tc], dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)
            danger_prob = probs[0][1].item()
            print(f"Input {tc} -> Danger Prob: {danger_prob:.4f}")

    # 3. Export to ONNX
    if not os.path.exists('models'):
        os.makedirs('models')
        
    dummy_input = torch.randn(1, 4)
    output_path = "models/signal_danger.onnx"
    
    print(f"\nExporting to {output_path}...")
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Done!")

if __name__ == "__main__":
    main()
