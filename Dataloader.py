"""
this file is used to:

1. define the dataset structure used in training or testing.
2. read the preprocessed data named "data.json", this file is a dictionary in shape {index: {"input": [beat_num, 1, 128, 87],"label": [beat_num, 256]}}

for example: to get a sample from data.json, we can use `data[index]["input"]` and `data[index]["label"]`
"""
import torch
import torch.utils.data
from pathlib import Path
import tempfile

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path="dataset_tensors"):
        """ Lazily loads compiled chunks targeting the OS Temporary directory natively """
        super(Dataset, self).__init__()
        
        target_path = Path(tempfile.gettempdir()) / "audio2chart_tensors"
        self.files = list(target_path.glob("*.pt"))
        if not self.files:
            print(f"Warning: No compiled training chunks discovered inside {target_path}!")

    def __getitem__(self, index):
        # Dynamically grabs and drops tensor arrays instantly mapping memory conservatively
        return torch.load(self.files[index], weights_only=True)

    def __len__(self):
        return len(self.files)
