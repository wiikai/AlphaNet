import torch
import torch.optim as optim
from model import AlphaNetV1
import torch.nn.functional as F
from preprocess import create_dynamic_dataloader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def train_model(train_dataloader, val_dataloader, model, optimizer, epochs=50, early_stopping=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    patience_counter = 0
    best_loss = np.inf
    train_losses = []
    val_losses = []
    best_model = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for idx, (images, labels, _) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = F.mse_loss(outputs, labels)
            # if torch.isnan(loss).any():
            #     print(f"Loss is NaN at epoch {epoch}, batch {idx}")
            #     break
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if idx % 50 == 0:
                print('Current epoch: %d, Current train batch: %d, Loss is %.6f' %(epoch+1,idx+1,loss.item()))

        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)

        val_loss = test_model(val_dataloader, model, epoch=epoch)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print("Early stopping")
            break
    return train_losses, val_losses, best_model

def test_model(dataloader, model, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = F.mse_loss(outputs, labels)
            total_loss += loss.item()
            if idx % 50 == 0:
                print('Current epoch: %d, Current vali batch: %d, Loss is %.6f' %(epoch+1,idx+1,loss.item()))

    return total_loss / len(dataloader)

def rolling_train(data, window_size=504, train_size=252, val_size=252, step=63, gap=5, epochs=50, early_stopping=10, data_window=30):
    data_slice = data_window - 1
    levels = data.index.get_level_values('date').unique()
    num_features = len(data.columns) -1
    max_index = len(levels) - window_size - gap * 2 - step - data_window + 1
    i = 0
    res = pd.DataFrame()

    while i <= max_index:
        print(f"Rolling Window: {i}/{max_index}")

        window_dates = levels[i:i + window_size + gap * 2 + step + data_slice]
        train_dates = window_dates[:train_size + data_slice]
        val_dates = window_dates[train_size + gap:train_size + gap + val_size + data_slice]
        test_dates = window_dates[train_size + gap * 2 + val_size:train_size + gap * 2 + val_size + step + data_slice]

        train_data = data.loc[train_dates]
        val_data = data.loc[val_dates]
        test_data = data.loc[test_dates]

        train_dataloader = create_dynamic_dataloader(train_data, window=data_window, step=1, shuffle=True)
        val_dataloader = create_dynamic_dataloader(val_data, window=data_window, step=1, shuffle=True)
        test_dataloader = create_dynamic_dataloader(test_data, window=data_window, step=1, shuffle=False)

        # for batch_images, batch_labels, nonrealize in train_dataloader:
        #     print(f'Batch images shape: {batch_images.shape}') 
        #     print(f'Batch labels shape: {batch_labels.shape}')
        #     print(f'Batch images[0]: {batch_images[0]}') 
        #     print(f'Batch batch_labels[0]: {batch_labels[0]}') 
        #     print(f'Batch nonrealize length: {len(nonrealize)}') 
        #     print(f'Batch nonrealize: {nonrealize}') 
        #     break 
    
        model = AlphaNetV1(num_features=num_features, size=10, rolling=21)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_losses, val_losses, best_model = train_model(train_dataloader, val_dataloader, model, optimizer, epochs=epochs, early_stopping=early_stopping)
        torch.save(best_model, f'./test/{test_dates[-step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")},({round(train_losses[val_losses.index(min(val_losses))],3)},{round(min(val_losses),3)}).pth')

        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss') 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./test/learning_curve_{test_dates[-step:][0].strftime("%Y%m%d")}-{test_dates[-1].strftime("%Y%m%d")}.png')
        plt.close()

        model.load_state_dict(best_model)
        model.eval()
        predictions = pd.Series(index=test_data.loc[test_dates[-step:]].index)
        with torch.no_grad():
            for idx, (images, labels, nonrealize) in enumerate(test_dataloader):
                outputs = model(images).squeeze().numpy()
                predictions.loc[test_dates[-step:][idx]][~predictions.loc[test_dates[-step:][idx]].index.isin(nonrealize)] = outputs
        
        res = pd.concat([res, predictions.unstack('order_book_id')], axis=0)
        i += step
    res.to_csv('./test/Factors.xlsx')

def main():
    set_seed(0)
    data = pd.read_parquet('test.parquet')
    rolling_train(data)

if __name__ == '__main__':
    main()