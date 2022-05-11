from pathlib import Path
import os

import random
import math
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from collections import Counter
from itertools import accumulate

import pdb

CLASSES = [1,2,3,4,5]

PERTURBATION={'speed': (lambda x: ['speed', f"{x}"]),
             'trim': (lambda x: ['trim', "0", f"{-x}"]), 
             'pad': (lambda x: ['pad', "0", f"{x}"]),
             'tempo': (lambda x: ['tempo', f"{x}"]),
             'pitch': (lambda x: ['pitch', f"{x}"]),
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

    for i in range(length):
        perturb = PERTURBATION[perturb_type](perturb_ratio)

        apply_effect_file_list.append([
            ["channels", "1"],
            ["rate", "16000"],
            perturb,
            ["norm"],
        ])
    
    return apply_effect_file_list


class VoiceMOSDataset(Dataset):
    def __init__(self, mos_list, ld_score_list, wav_folder, corpus_name, perturb_mode='none', perturb_types=[], perturb_ratios=[], total_length=-1, valid=False):
        self.wav_folder = Path(wav_folder)
        self.mos_list = mos_list
        self.ld_score_list = ld_score_list
        self.corpus_name = corpus_name

        self.perturb_mode = perturb_mode
        self.perturb_types = perturb_types
        self.perturb_ratios = perturb_ratios
        self.apply_effect_file_list = []

        self.class_num = 5
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self._JUDGE = 4

        self.valid = valid

        self.total_length = total_length if (total_length != -1) else len(self.mos_list)

        # generate list of effects for apply_effect_file()
        assert self.perturb_mode in PERTURBATION_MODE, "Invalid perturbation mode"
            
        self.apply_effect_file_list += generate_apply_effect_file_commands(self.total_length)
        
        if self.perturb_mode == 'fixed':
            for perturb_type, perturb_ratio in zip(self.perturb_types, self.perturb_ratios):
                self.apply_effect_file_list += generate_apply_effect_file_commands((self.total_length), perturb_type=perturb_type, perturb_ratio=perturb_ratio)

        print(f"[Dataset Information] - MOS Score dataset \'{corpus_name}\' using perturbation type \'{perturb_mode}\'. Dataset length={len(self.apply_effect_file_list)}")

    def __len__(self):
        return len(self.apply_effect_file_list)

    def __getitem__(self, idx):
        list_idx = idx % self.total_length % len(self.mos_list)

        wav_name, mos = self.mos_list.loc[list_idx]
        wav_path = self.wav_folder / wav_name
        effects = self.apply_effect_file_list[idx]

        if self.perturb_mode == 'random':
            perturb_type, perturb_ratio = random.choice(list(zip(self.perturb_types, self.perturb_ratios)))
            effects = generate_apply_effect_file_commands(1, perturb_type=perturb_type, perturb_ratio=perturb_ratio)[0]

        wav, _ = apply_effects_file(
            str(wav_path),
            effects
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]
        corpus_name = self.corpus_name
        judge_id = 0
        prob = np.zeros(self.class_num)

        # If not in validation, then probability is needed
        if self.valid == False: 
            wav_ld_score_list = list(self.ld_score_list[self.ld_score_list[1] == wav_name][2])
            wav_ld_score_index_list = [self.class2index[score] for score in wav_ld_score_list]
            wav_ld_score_counter = np.zeros(self.class_num)
            
            for index, value in Counter(wav_ld_score_index_list).items():
                wav_ld_score_counter[index] = value
            
            prob = wav_ld_score_counter / np.sum(wav_ld_score_counter)

        return wav.numpy(), system_name, wav_name, corpus_name, mos, prob, judge_id


    def collate_fn(self, samples):
        return zip(*samples)
            


class VoiceMOSLDScoreDataset(Dataset):
    def __init__(self, ld_score_list, wav_folder, corpus_name, perturb_mode='none', perturb_types=[], perturb_ratios=[], idtable=''):
        self.wav_folder = Path(wav_folder)
        self.ld_score_list = ld_score_list
        self.corpus_name = corpus_name

        self.perturb_mode = perturb_mode
        self.perturb_types = perturb_types
        self.perturb_ratios = perturb_ratios
        self.apply_effect_file_list = []

        self.class_num = 5
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self._JUDGE = 4

        self.total_length = len(self.ld_score_list)

        # generate list of effects for apply_effect_file()
        assert self.perturb_mode in PERTURBATION_MODE, "Invalid perturbation mode"
            
        self.apply_effect_file_list += generate_apply_effect_file_commands(self.total_length)
        
        if self.perturb_mode == 'fixed':
            for perturb_type, perturb_ratio in zip(self.perturb_types, self.perturb_ratios):
                self.apply_effect_file_list += generate_apply_effect_file_commands((self.total_length), perturb_type=perturb_type, perturb_ratio=perturb_ratio)

        print(f"[Dataset Information] - Listener Dependent Score dataset \'{corpus_name}\' using perturbation type \'{perturb_mode}\'. Dataset length={len(self.apply_effect_file_list)}")

        # Load idtable
        assert Path.is_file(idtable), f"Can't find idtable file: {idtable}"

        self.idtable = torch.load(idtable)
        for i, judge_i in enumerate(self.ld_score_list[self._JUDGE]):
            self.ld_score_list[self._JUDGE][i] = self.idtable[judge_i]

    def __len__(self):
        return len(self.apply_effect_file_list)

    def __getitem__(self, idx):
        list_idx = idx % self.total_length
        system_name, wav_name, opinion_score, _, judge_id = self.ld_score_list.loc[list_idx]

        wav_path = self.wav_folder / wav_name
        effects = self.apply_effect_file_list[idx]

        if self.perturb_mode == 'random':
            perturb_type, perturb_ratio = random.choice(list(zip(self.perturb_types, self.perturb_ratios)))
            effects = generate_apply_effect_file_commands(1, perturb_type=perturb_type, perturb_ratio=perturb_ratio)[0]

        wav, _ = apply_effects_file(
            str(wav_path),
            effects
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]
        corpus_name = self.corpus_name
        prob = np.zeros(self.class_num)
        prob[self.class2index[opinion_score]] = 1

        return wav.numpy(), system_name, wav_name, corpus_name, opinion_score, prob, judge_id


    def collate_fn(self, samples):
        return zip(*samples)
            