from torch.utils.data import Dataset
import torch


class SkipGramDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = [(word_to_ix[center], word_to_ix[context]) for center, context in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.long), torch.tensor(self.data[idx][1], dtype=torch.long)
