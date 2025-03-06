from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, samples):
        """
        A custom PyTorch Dataset to handle the samples and tokenize them.
        """
        self.samples = samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Tokenize and return a single sample with truncation.
        """
        sample = self.samples[idx]
        return sample  # return the tokens and the original sample for future processing

