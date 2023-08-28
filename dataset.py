from torch.utils.data import Dataset
import torch
from constant import START_TOKEN, END_TOKEN
from torch.nn.utils.rnn import pad_sequence
mps_device = torch.device("mps")

class SenCLSDataset(Dataset):
    def __init__(self, datas, vocab):
        self.preprocess(datas, vocab)

    def preprocess(self, datas, vocab):
        self.x, self.y = [], []
        for data in datas:
            sent, label = data

            
            if isinstance(sent, str):  # Check if 'sent' is a string
                words = sent.split()
                self.x.append([vocab[w] for w in words if w in vocab])  # Only include words in vocab
            else:
                self.x.append([])  # Add empty list for non-string sentences
            self.y.append(label)

        max_len = 64  
        self.x = [torch.tensor(seq + [0]*(max_len - len(seq))) if len(seq) < max_len else torch.tensor(seq[:max_len]) for seq in self.x] 
        # self.x = pad_sequence(self.x, batch_first=True, padding_value=0)
        # Convert self.y to a tensor
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


