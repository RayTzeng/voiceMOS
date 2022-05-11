import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from tqdm import tqdm
import pdb

from .dataset import BVCCDataset, NISQADataset
from .model import Model

warnings.filterwarnings("ignore")

class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = kwargs["expdir"]
        self.BVCC_config = self.datarc["BVCC"]
        self.NISQA_config = self.datarc["NISQA"]

        self.train_dataset = []
        self.eval_dataset = []
        self.test_dataset = []
        self.best = {}

        # Build Training Dataset
        for dataset_name in self.datarc["train_dataset"]:
            assert (dataset_name == "BVCC" or dataset_name == "NISQA"), "Invalid Dataset Choice"

            if dataset_name == "BVCC":
                base_path = self.BVCC_config["base_path"]
                csv_file = Path(base_path) / self.BVCC_config["train_csv"]
                dataset = BVCCDataset(csv_file, base_path, "train")
                self.train_dataset.append(dataset)
            
            elif dataset_name == "NISQA":
                base_path = self.NISQA_config["base_path"]
                csv_file = Path(base_path) / self.NISQA_config["csv_file"]
                for split in self.NISQA_config["train_split"]:
                    dataset = NISQADataset(csv_file, base_path, split)
                    self.train_dataset.append(dataset)
        
        # Build Evaluation Dataset
        for dataset_name in self.datarc["eval_dataset"]:
            assert (dataset_name == "BVCC" or dataset_name == "NISQA"), "Invalid Dataset Choice"

            if dataset_name == "BVCC":
                base_path = self.BVCC_config["base_path"]
                csv_file = Path(base_path) / self.BVCC_config["eval_csv"]
                dataset = BVCCDataset(csv_file, base_path, "eval")
                self.eval_dataset.append(dataset)
            
            elif dataset_name == "NISQA":
                base_path = self.NISQA_config["base_path"]
                csv_file = Path(base_path) / self.NISQA_config["csv_file"]
                for split in self.NISQA_config["eval_split"]:
                    dataset = NISQADataset(csv_file, base_path, split)
                    self.eval_dataset.append(dataset)

        # Build Test Dataset
        for dataset_name in self.datarc["test_dataset"]:
            assert (dataset_name == "BVCC" or dataset_name == "NISQA"), "Invalid Dataset Choice"

            if dataset_name == "BVCC":
                base_path = self.BVCC_config["base_path"]
                csv_file = Path(base_path) / self.BVCC_config["test_csv"]
                dataset = BVCCDataset(csv_file, base_path, "test")
                self.test_dataset.append(dataset)
            
            elif dataset_name == "NISQA":
                base_path = self.NISQA_config["base_path"]
                csv_file = Path(base_path) / self.NISQA_config["csv_file"]
                for split in self.NISQA_config["test_split"]:
                    dataset = NISQADataset(csv_file, base_path, split)
                    self.test_dataset.append(dataset)
        
        self.collate_fn = self.train_dataset[0].collate_fn

        # Define Evaluation Metrics
        if Path(self.expdir, "best.pkl").is_file():
            with open(Path(self.expdir, "best.pkl"), "r") as f:
                self.best = json.load(f)
            print("Found existing best score records")
            print(self.best)
        else:
            self.best = {
                "step": 0,
                "eval_loss": 10000
            }

        # build downstream models
        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_size=self.modelrc["projector_dim"],
            pooling_name=self.modelrc["pooling_name"],
            dim=self.modelrc["dim"],
            dropout=self.modelrc["dropout"],
            activation=self.modelrc["activation"]
        )

        print('[Model Information] - Printing downstream model information')
        print(self.model)

        self.objective = eval(f"nn.{self.modelrc['objective']}")()

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(ConcatDataset(self.train_dataset))
        elif mode == "eval":
            return self._get_eval_dataloader(ConcatDataset(self.eval_dataset))
        elif mode == "test":
            return self._get_eval_dataloader(ConcatDataset(self.test_dataset))



    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.collate_fn,
        )

    # Interface
    def forward(
        self,
        mode,
        features,
        mos_list,
        corpus_name_list,
        system_name_list,
        wav_name_list,
        records,
        **kwargs,
    ):

        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)

        mos_list = torch.FloatTensor(mos_list).to(features.device)

        preds = self.model(features, features_len)

        if mode == "train" or mode == "eval":

            loss = self.objective(preds, mos_list)
            records["loss"].append(loss.item())

        mos_list = mos_list.detach().cpu().tolist()
        pred_list = preds.detach().cpu().tolist()

        records["corpus_name"] += corpus_name_list
        records["system_name"] += system_name_list
        records["wav_name"] += wav_name_list
        records["pred"] += pred_list
        records["mos"] += mos_list

        if mode == "train":
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss
        if mode == "train" or mode == "eval":
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(
                f"Loss/{mode}",
                avg_loss,
                global_step=global_step,
            )

            # save checkpoint
            if mode == "eval":
                if avg_loss < self.best["eval_loss"]:
                    self.best["eval_loss"] = avg_loss
                    self.best["step"] = global_step

                    with open(Path(self.expdir, "best.pkl"), "w") as f:
                        json.dump(self.best, f)
                    save_names.append("best.ckpt")


        if mode == "eval" or mode == "test":
            sys_pred_score = defaultdict(lambda: defaultdict(list))
            sys_true_score = defaultdict(lambda: defaultdict(list))

            uttr_pred_score = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            uttr_true_score = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

            for corpus_name, system_name, wav_name, pred, mos in zip(records["corpus_name"], records["system_name"], records["wav_name"], records["pred"], records["mos"]):
                sys_pred_score[corpus_name][system_name].append(pred)
                sys_true_score[corpus_name][system_name].append(mos)

                uttr_pred_score[corpus_name][system_name][wav_name] = pred
                uttr_true_score[corpus_name][system_name][wav_name] = mos

            for corpus_name in uttr_true_score.keys():
                # BVCC Corpus Logging
                if corpus_name == "BVCC":
                    sys_pred_score_list = []
                    sys_true_score_list = []
                    wav_name_list = []
                    uttr_pred_score_list = []
                    uttr_true_score_list = []

                    for system_name in uttr_true_score[corpus_name].keys():
                        sys_pred_score_list.append(np.mean(sys_pred_score[corpus_name][system_name]))
                        sys_true_score_list.append(np.mean(sys_true_score[corpus_name][system_name]))

                        for wav_name in uttr_true_score[corpus_name][system_name].keys():
                            wav_name_list.append(wav_name)
                            uttr_pred_score_list.append(uttr_pred_score[corpus_name][system_name][wav_name])
                            uttr_true_score_list.append(uttr_true_score[corpus_name][system_name][wav_name])

                    MSE, LCC, SRCC, KTAU = self.metrics(np.array(sys_true_score_list), np.array(sys_pred_score_list))
                    for metric in ['MSE', 'LCC', 'SRCC', 'KTAU']:
                        logger.add_scalar(
                            f"{corpus_name}-{mode}/System-level-{metric}",
                            eval(metric),
                            global_step=global_step,
                        )
                        if self.best["step"] == global_step:
                            tqdm.write(f"[{corpus_name}] System-level {metric} = {eval(metric):.4f}")

                    MSE, LCC, SRCC, KTAU = self.metrics(np.array(uttr_true_score_list), np.array(uttr_pred_score_list))
                    for metric in ['MSE', 'LCC', 'SRCC', 'KTAU']:
                        logger.add_scalar(
                            f"{corpus_name}-{mode}/Utterance-level-{metric}",
                            eval(metric),
                            global_step=global_step,
                        )
                        if self.best["step"] == global_step:
                            tqdm.write(f"[{corpus_name}] Utterance-level {metric} = {eval(metric):.4f}")

                    # Save Predictions
                    if self.best["step"] == global_step:
                        df = pd.DataFrame(list(zip(wav_name_list, uttr_pred_score_list)))
                        os.makedirs(Path(self.expdir, corpus_name), exist_ok=True)
                        df.to_csv(Path(self.expdir, corpus_name, f"best_step_{mode}.txt"), header=None, index=None)
                
                # NISQA Corpus Logging
                elif corpus_name == "NISQA":
                    for system_name in uttr_true_score[corpus_name].keys():
                        wav_name_list = []
                        uttr_pred_score_list = []
                        uttr_true_score_list = []
                        for wav_name in uttr_true_score[corpus_name][system_name].keys():
                            wav_name_list.append(wav_name)
                            uttr_pred_score_list.append(uttr_pred_score[corpus_name][system_name][wav_name])
                            uttr_true_score_list.append(uttr_true_score[corpus_name][system_name][wav_name])

                        MSE, LCC, SRCC, KTAU = self.metrics(np.array(uttr_true_score_list), np.array(uttr_pred_score_list))
                        for metric in ['MSE', 'LCC', 'SRCC', 'KTAU']:
                            logger.add_scalar(
                                f"{corpus_name}-{system_name}/Utterance-level-{metric}",
                                eval(metric),
                                global_step=global_step,
                            )
                            if self.best["step"] == global_step:
                                tqdm.write(f"[{system_name}] Utterance-level {metric} = {eval(metric):.4f}")

                        # Save Predictions
                        if self.best["step"] == global_step:
                            df = pd.DataFrame(list(zip(wav_name_list, uttr_pred_score_list)))
                            os.makedirs(Path(self.expdir, corpus_name, system_name), exist_ok=True)
                            df.to_csv(Path(self.expdir, corpus_name, system_name, f"best_step_{mode}.txt"), header=None, index=None)

             
                    
                        

        return save_names

    def metrics(self, true, pred):
        MSE = np.mean((true - pred) ** 2)
        LCC = pearsonr(true, pred)[0]
        SRCC = spearmanr(true.T, pred.T)[0]
        KTAU = kendalltau(true, pred)[0]

        return MSE, LCC, SRCC, KTAU

def load_file(base_path, file):
    dataframe = pd.read_csv(Path(base_path, file), header=None)
    return dataframe
