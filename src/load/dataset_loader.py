import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['answer']
        label = self.df.iloc[idx]['generated']
        question = self.df.iloc[idx]['question']

        return {'input_ids': text, 'label': label, 'question': question}
