from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from model import ChartNet
from Dataloader import Dataset
from tqdm import tqdm
import argparse
import os

def train_model(data_path="data.pkl", epochs=30, batch_size=4, lr=0.001):
    # Dataloader hasn't explicitly changed structurally regarding keys: input/label
    print(f"Loading Dataset from {data_path}...")
    data_set = Dataset(data_path)
    train_len = int(len(data_set) * 0.8)
    train_set, test_set = random_split(data_set, [train_len, len(data_set) - train_len])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Booting training on: {device}")

    # Standard architecture mapped in Phase 3
    model = ChartNet(fc_feature=600, audio_feature=500, hidden_dim=512, num_layers=2, output_dim=5).to(device)

    # Multi-label classification requires BCEWithLogitsLoss logic
    # Adding a positive weight to heavily combat the sparse 'zeros' in drums
    pos_weight = torch.tensor([50.0, 50.0, 50.0, 50.0, 50.0], device=device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    start_epoch = 0
    checkpoint_path = "checkpoints/autosave.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming at epoch {start_epoch + 1}")

    print("Beginning Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        for index, sample in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = sample["input"].to(device) # Shape: [batch, seq_len, 1, 128, 87]
            labels = sample["label"].to(device) # Shape: [batch, seq_len, 5]
            
            optim.zero_grad()
            outputs = model(inputs) # Shape: [batch, seq_len, 5]
            
            # Loss expects [N, *] flattening
            loss = criterion(outputs.view(-1, 5), labels.view(-1, 5))
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            
        print(f"--- Epoch {epoch+1} Complete | Avg Loss: {running_loss/(index+1):.4f}")
        
        # Save checkpoints aggressively
        torch.save(model.state_dict(), f"checkpoints/drum_model_epoch_{epoch+1}.pth")
        
        # Save autosave checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, checkpoint_path)

    print("Training finished perfectly!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.pkl", help="Preprocessed data mapping.")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    train_model(data_path=args.data, epochs=args.epochs)
