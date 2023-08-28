import torch.nn as nn
import torch
from dataset import SenCLSDataset
from torch.utils.data import DataLoader
from model import LSTM
from vocab import get_vocab
from ds import *
from torch.utils.data import random_split

mps_device = torch.device("mps")

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(mps_device)
            y = y.to(mps_device)

            outputs = model(x)
            loss = criterion(outputs.squeeze(), y.float())
            total_loss += loss.item()

            
            predictions = torch.round(torch.sigmoid(outputs)).squeeze().long()
            correct = (predictions == y).sum().item()
            total_correct += correct
            total_samples += x.size(0)
            
            print(f"x {x}:")
            print(f"Batch {i}:")
            print(f"Outputs: {outputs}")
            print(f"Predictions: {predictions}")
            print(f"Ground truth: {y}")


    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def main():
    # load the vocab and the test data
    vocab = get_vocab()
    dataset = ds()
    # assume you have loaded or created your test data in the same format as your training data
    x_test, y_test = dataset.get_test() # implement this function based on your test data
    test_data = list(zip(x_test, y_test))
    test_dataset = SenCLSDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=10)

    # load the model
    model = LSTM(len(vocab), 300, 128, 2, 0.5)
    model.load_state_dict(torch.load('./best_model.pth'))
    model.to(mps_device)

    # evaluate the model on the test set
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
