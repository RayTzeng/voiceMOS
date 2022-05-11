from pathlib import Path
import os

import random
import math
import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from collections import Counter
from itertools import accumulate

import pdb

class BVCCDataset(Dataset):
    def __init__(self, csv_file, base_path, split):
        self.base_path = Path(base_path)
        self.wav_list = self.get_wav_list(csv_file, split)
        self.corpus_name = "BVCC"

        print(f"[Dataset Information] - BVCC dataset. Dataset length={len(self.wav_list)}")

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name, mos = self.wav_list.loc[idx]
        
        wav_path = self.base_path/ "wav" / wav_name

        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ]
        )

        wav = wav.view(-1)
        corpus_name = self.corpus_name
        system_name = wav_name.split("-")[0]

        return wav.numpy(), mos, corpus_name, system_name, wav_name

    def collate_fn(self, samples):
        return zip(*samples)

    def get_wav_list(self, csv_file, split):
        df = self.load_file(csv_file)
        wav_list = df
        return wav_list
        
    def load_file(self, file):
        dataframe = pd.read_csv(Path(file), header=None)
        return dataframe

class NISQADataset(Dataset):
    def __init__(self, csv_file, base_path, split):
        self.base_path = Path(base_path)
        self.wav_list = self.get_wav_list(csv_file, split)
        self.corpus_name = "NISQA"
        self.split = split

        print(f"[Dataset Information] - NISQA dataset. Dataset length={len(self.wav_list)}")

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name, mos = self.wav_list.loc[idx]

        wav_path = self.base_path / wav_name

        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ]
        )

        wav = wav.view(-1)
        corpus_name = self.corpus_name
        system_name = self.split
        
        return wav.numpy(), mos, corpus_name, system_name, wav_name

    def collate_fn(self, samples):
        return zip(*samples)

    def get_wav_list(self, csv_file, split):
        df = self.load_file(csv_file)
        wav_list = df.loc[df.db==split, ["filepath_deg", "mos"]].reset_index(drop=True)
        return wav_list
        
    def load_file(self, file):
        dataframe = pd.read_csv(Path(file))
        return dataframe