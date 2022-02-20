from pathlib import Path
import os

import random
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

import pdb

PERTURBATION={'speed_up': (lambda x: ['speed', f"{1.0 + x}"]),
             'speed_down': (lambda x: ['speed', f"{1.0 - x}"]), 
             'trim': (lambda x: ['trim', "0", f"{-x}"]), 
             'pad': (lambda x: ['pad', "0", f"{x}"])
             }

PERTURBATION_MODE=['none', 'fixed', 'random']

def generate_apply_effect_file_commands(length, perturb_type='none', perturb_ratio=None):
    apply_effect_file_list = []

    if perturb_type == 'none':
        for i in range(length):
            apply_effect_file_list.append([
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ])

        return apply_effect_file_list

    assert perturb_type in list(PERTURBATION.keys()), "Invalid perturbation type."
    
    x_range = np.linspace(0,perturb_ratio,101)

    for i in range(length):
        x = random.choice(x_range)
        perturb = PERTURBATION[perturb_type](x)

        apply_effect_file_list.append([
            ["channels", "1"],
            ["rate", "16000"],
            perturb,
            ["norm"],
        ])
    
    return apply_effect_file_list


class VoiceMOSDataset(Dataset):
    def __init__(self, mos_list, wav_folder, corpus_name, perturb_mode='none', perturb_types=[], perturb_ratios=[]):
        self.wav_folder = Path(wav_folder)
        self.mos_list = mos_list
        self.corpus_name = corpus_name
        self.perturb_mode = perturb_mode
        self.perturb_types = perturb_types
        self.perturb_ratios = perturb_ratios
        self.apply_effect_file_list = []

        assert self.perturb_mode in PERTURBATION_MODE, "Invalid perturbation mode"
            
        self.apply_effect_file_list += generate_apply_effect_file_commands(len(self.mos_list))
        
        if self.perturb_mode == 'fixed':
            for perturb_type, perturb_ratio in zip(self.perturb_types, self.perturb_ratios):
                self.apply_effect_file_list += generate_apply_effect_file_commands(len(self.mos_list), perturb_type=perturb_type, perturb_ratio=perturb_ratio)

    def __len__(self):
        return len(self.apply_effect_file_list)

    def __getitem__(self, idx):
        wav_name, mos = self.mos_list.loc[idx % len(self.mos_list)]
        wav_path = self.wav_folder / wav_name
        effects = self.apply_effect_file_list[idx]

        if self.perturb_mode == 'random':
            perturb_type, perturb_ratio = random.choice(list(zip(self.perturb_types, self.perturb_ratios)))
            effects = generate_apply_effect_file_commands(1, perturb_type=perturb_type, pertrub_ratio=pertrub_ratio)[0]

        wav, _ = apply_effects_file(
            str(wav_path),
            effects
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]
        corpus_name = self.corpus_name

        return wav.numpy(), system_name, wav_name, corpus_name, mos


    def collate_fn(self, samples):
        return zip(*samples)
            
    