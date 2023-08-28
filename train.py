import torch.nn as nn
import torch
from dataset import SenCLSDataset
from torch.utils.data import DataLoader
from model import LSTM
from vocab import get_vocab
from ds import *
from torch.utils.data import random_split

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
mps_device = torch.device("mps")
dataset = ds()
x_train, y_train = dataset.get_train()
data = list(zip(x_train, y_train))

vocab = get_vocab()
dataset = SenCLSDataset(data, vocab)
# loader = DataLoader(dataset, batch_size=10)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = LSTM (len(vocab), 300,128, 2, 0.5)
model.to(mps_device)

optimizer = torch.optim.Adam(model.parameters(), lr =1e-3 )
criterion = nn.BCEWithLogitsLoss()


best_val_loss = float('inf')
patience = 3
counter = 0
nb_epochs = 5

for epoch in range(nb_epochs+1):
    print('Epoch [{}/{}]'.format(epoch + 1, nb_epochs))
    avg_cost = 0

    model.train()
    for i, (x,y) in enumerate(train_loader):  #batch 로 받는걸 x,y 로 나누어줌
        # y= y.float()
        x = x.to(mps_device)
        y = y.to(mps_device)
        outputs = model(x)
        cost = criterion(outputs.squeeze(), y.float())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # print(f"x : {x}")
        # print(f"Outputs: {outputs}")
        # print(f"Ground truth: {y}")
        
        print('Train Epoch [{}/{}], idx {:4d}, Cost: {:.6f}'.format(epoch + 1, nb_epochs, i, cost))
        
        
    # if epoch % 10 == 0:
    #     # 10번마다 로그 출력
    #     print('Epoch {:4d}/{} Cost: {:.6f}'.format(
    #       epoch, nb_epochs, cost.item()
    #   ))
        
    model.eval()
    val_loss = 0 
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(mps_device)
            y = y.to(mps_device)
            outputs = model(x)
            cost = criterion(outputs.squeeze(), y.float())
            val_loss += cost.item()

    val_loss /= len(val_loader)
    print('Val Epoch [{}/{}], Validation Loss: {:.6f}'.format(epoch + 1, nb_epochs, val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './best_model.pth')
        print('Model saved at epoch {}, with Validation Loss: {:.6f}'.format(epoch + 1, val_loss))
        counter = 0
    else:
        counter += 1
        print('No improvement in validation loss for {} epoch(s)'.format(counter))
        if counter >= patience:
            print('Early stopping triggered.')
            break
        


