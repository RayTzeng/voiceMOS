import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from tqdm import tqdm
import pdb

from .dataset import VoiceMOSDataset, VoiceMOSLDScoreDataset
from .model import Model

warnings.filterwarnings("ignore")

TRUE_SCORE_IDX=0
PRED_SCORE_IDX=1
WAV_NAME_IDX=2

JUDGE=4
SCORE=2


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = kwargs["expdir"]


        if Path(kwargs["expdir"], "idtable.pkl").is_file():
            idtable_path = str(Path(kwargs["expdir"], "idtable.pkl"))
            print(f"[Dataset Information] - Found existing idtable at {idtable_path}")
            self.idtable = Path(idtable_path)
        elif Path(self.datarc['idtable']).is_file():
            print(f"[Dataset Information] - Found existing idtable at {self.datarc['idtable']}")
            self.idtable = Path(self.datarc['idtable'])
        else:
            print(f"[Dataset Information] - Generate new idtable")
            self.idtable = Path(kwargs["expdir"]) / "idtable.pkl"
            self.gen_idtable(self.idtable)

        # Generate or load idtable

        self.train_dataset = []
        self.train_eval_dataset = []
        self.dev_dataset = []
        self.test_dataset = []
        self.system_mos = {}
        self.best_scores = {}
        self.record_names = ['mean_score', 'reg_score', 'class_score']

        print(f"[Dataset Information] - Using dataset {self.datarc['corpus_names']}")

        for i, (data_folder, corpus_name) in enumerate(zip(self.datarc['data_folders'], self.datarc['corpus_names'])):
            perturbrc = self.datarc['perturb']

            print(f"[Dataset Information] - [Train split]")
            train_mos_df = load_file(data_folder, self.datarc["train_mos_list_path"])
            train_ld_score_df = load_file(data_folder, self.datarc["train_ld_score_list_path"])
            train_wav_folder = Path(data_folder) / 'wav'
            train_mos_length = len(train_ld_score_df) if self.datarc["ld_score_bool"] else -1

            self.train_dataset.append(VoiceMOSDataset(mos_list=train_mos_df, 
                                                    ld_score_list=train_ld_score_df,
                                                    wav_folder=train_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    perturb_mode=perturbrc["mode"], 
                                                    perturb_types=perturbrc["types"], 
                                                    perturb_ratios=perturbrc["ratios"],
                                                    total_length=train_mos_length
                                                    ))

            if self.datarc["ld_score_bool"]:
                self.train_dataset.append(VoiceMOSLDScoreDataset(ld_score_list=train_ld_score_df, 
                                                        wav_folder=train_wav_folder, 
                                                        corpus_name=corpus_name, 
                                                        perturb_mode=perturbrc["mode"], 
                                                        perturb_types=perturbrc["types"], 
                                                        perturb_ratios=perturbrc["ratios"],
                                                        idtable=self.idtable
                                                        ))

            self.train_eval_dataset.append(VoiceMOSDataset(mos_list=train_mos_df, 
                                                    ld_score_list=train_ld_score_df,
                                                    wav_folder=train_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    ))

            print(f"[Dataset Information] - [Valid split]")
            valid_mos_df = load_file(data_folder, self.datarc["val_mos_list_path"])
            valid_ld_score_df = load_file(data_folder, self.datarc["val_ld_score_list_path"])
            valid_wav_folder = Path(data_folder) / 'wav'

            self.dev_dataset.append(VoiceMOSDataset(mos_list=valid_mos_df, 
                                                    ld_score_list=valid_ld_score_df,
                                                    wav_folder=valid_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    ))

            print(f"[Dataset Information] - [Test split]")
            test_mos_df = load_file(data_folder, self.datarc["test_mos_list_path"])
            test_wav_folder = self.datarc['test_wav_folders'][i] if (len(self.datarc['test_wav_folders']) == len(self.datarc['data_folders'])) else (Path(data_folder) / 'wav')

            self.test_dataset.append(VoiceMOSDataset(mos_list=test_mos_df, 
                                                    ld_score_list=None,
                                                    wav_folder=test_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    valid=True
                                                    ))

            self.system_mos[corpus_name] = pd.read_csv(Path(data_folder, "system_level_mos.csv"), index_col=False)
            self.best_scores[corpus_name] = {
                "MSE": 10000,
                "LCC": -1.0,
                "SRCC": -1.0,
            }
        
        self.collate_fn = self.train_dataset[0].collate_fn

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_size=self.modelrc["projector_dim"],
            regression_output_size=1,
            classification_output_size=5,
            pooling_name=self.modelrc["pooling_name"],
            dim=self.modelrc["dim"],
            dropout=self.modelrc["dropout"],
            activation=self.modelrc["activation"]
        )

        print('[Model Information] - Printing downstream model information')
        print(self.model)

        scores = []
        for data_folder in self.datarc['data_folders']:
            ld_score_list = load_file(data_folder, self.datarc["train_ld_score_list_path"])
            scores += list(ld_score_list[SCORE])

        class_weights = self.calc_class_weight(scores)
        self.classification_objective = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
        self.classification_weight = self.modelrc['classification_weight']

        self.regression_objective = eval(f"nn.{self.modelrc['regression_objective']}")()
        self.regression_weight = self.modelrc['regression_weight']
        


    def gen_idtable(self, idtable_path):
        idtable = {}
        count = 1
        for data_folder in self.datarc['data_folders']:
            ld_score_list = load_file(data_folder, self.datarc["train_ld_score_list_path"])
            for i, judge_i in enumerate(ld_score_list[JUDGE]):
                if judge_i not in idtable.keys():
                    idtable[judge_i] = count
                    count += 1

        torch.save(idtable, idtable_path)
    
    def calc_class_weight(self, scores):
        class_weights = class_weight.compute_class_weight('balanced', classes=np.linspace(1,5,5),y=np.array(scores))
        return class_weights

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(ConcatDataset(self.train_dataset))
        elif mode == "train_eval":
            return self._get_eval_dataloader(ConcatDataset(self.train_eval_dataset))
        elif mode == "dev":
            return self._get_eval_dataloader(ConcatDataset(self.dev_dataset))
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
        system_name_list,
        wav_name_list,
        corpus_name_list,
        mos_list,
        prob_list,
        judge_id_list,
        records,
        **kwargs,
    ):
        # NOT YET


        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)

        prob_list = torch.FloatTensor(prob_list).to(features.device)
        mos_list = torch.FloatTensor(mos_list).to(features.device)
        judge_id_list = torch.LongTensor(judge_id_list).to(features.device)

        reg_scores, logits = self.model(features, features_len, judge_id_list)

        if mode == "train" or mode == "dev":
            reg_loss = self.regression_objective(reg_scores, mos_list)
            class_loss = self.classification_objective(logits, prob_list)
            loss = self.regression_weight * reg_loss + self.classification_weight * class_loss

            records["regression loss"].append(reg_loss.item())
            records["classification loss"].append(class_loss.item())
            records["total loss"].append(loss.item())

        class_scores = torch.matmul(F.softmax(logits, dim=1), torch.linspace(1,5,5).to(logits.device))
        true_scores = torch.matmul(prob_list, torch.linspace(1,5,5).to(prob_list.device))

        reg_scores = reg_scores.detach().cpu().tolist()
        class_scores = class_scores.detach().cpu().tolist()  
        mean_scores = (np.array(class_scores) + np.array(reg_scores)) / 2
        mos_list = mos_list.detach().cpu().tolist()

        for record_name, score_list in zip(self.record_names, [mean_scores, reg_scores, class_scores]):
            if len(records[record_name]) == 0:
                for _ in range(3):
                    records[record_name].append(defaultdict(lambda: defaultdict(list)))

            for corpus_name, system_name, wav_name, score, mos in zip(corpus_name_list, system_name_list, wav_name_list, score_list, mos_list):
                records[record_name][PRED_SCORE_IDX][corpus_name][system_name].append(score)
                records[record_name][TRUE_SCORE_IDX][corpus_name][system_name].append(mos)
                records[record_name][WAV_NAME_IDX][corpus_name][system_name].append(wav_name)

        if mode == "train":
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss

        if mode == "train" or mode == "dev":
            avg_total_loss = np.mean(records["total loss"])
            logger.add_scalar(
                f"Total-loss/{mode}",
                avg_total_loss,
                global_step=global_step,
            )

            avg_reg_loss = np.mean(records["regression loss"])
            logger.add_scalar(
                f"Regression-loss/{mode}",
                avg_reg_loss,
                global_step=global_step,
            )

            avg_class_loss = np.mean(records["classification loss"])
            logger.add_scalar(
                f"Classification-loss/{mode}",
                avg_class_loss,
                global_step=global_step,
            )

        # logging Utterance-level MSE, LCC, SRCC

        if mode == "train_eval" or mode == "dev":
            # some evaluation-only processing, eg. decoding
            for record_name in self.record_names:

                for corpus_name in self.datarc['corpus_names']:
                    corpus_pred_score_list = []
                    corpus_true_score_list = []
                    corpus_wav_name_list = []
                    corpus_system_pred_score_list = []
                    corpus_system_true_score_list = []

                    for system_name in list(records[record_name][TRUE_SCORE_IDX][corpus_name].keys()):
                        corpus_pred_score_list += records[record_name][PRED_SCORE_IDX][corpus_name][system_name]
                        corpus_true_score_list += records[record_name][TRUE_SCORE_IDX][corpus_name][system_name]
                        corpus_wav_name_list += records[record_name][WAV_NAME_IDX][corpus_name][system_name]
                        corpus_system_pred_score_list.append(np.mean(records[record_name][PRED_SCORE_IDX][corpus_name][system_name]))
                        corpus_system_true_score_list.append(np.mean(records[record_name][TRUE_SCORE_IDX][corpus_name][system_name]))
                    
                    # Calculate utterance level metric 
                    corpus_pred_scores = np.array(corpus_pred_score_list)
                    corpus_true_scores = np.array(corpus_true_score_list)

                    MSE = np.mean((corpus_true_scores - corpus_pred_scores) ** 2)
                    LCC, _ = pearsonr(corpus_true_scores, corpus_pred_scores)
                    SRCC, _ = spearmanr(corpus_true_scores.T, corpus_pred_scores.T)


                    for metric in ['MSE', 'LCC', 'SRCC']:
                        logger.add_scalar(
                            f"Utterance-level-{record_name}/{corpus_name}-{mode}-{metric}",
                            eval(metric),
                            global_step=global_step,
                        )

                        # tqdm.write(f"[{record_name}] [{corpus_name}] [{mode}] Utterance-level {metric}  = {eval(metric):.4f}")


                    # Calculate system level metric 
                    system_level_mos = self.system_mos[corpus_name]

                    corpus_system_pred_scores = np.array(corpus_system_pred_score_list)
                    corpus_system_true_scores = np.array(corpus_system_true_score_list)

                    MSE = np.mean((corpus_system_true_scores - corpus_system_pred_scores) ** 2)
                    LCC, _ = pearsonr(corpus_system_true_scores, corpus_system_pred_scores)
                    SRCC, _ = spearmanr(corpus_system_true_scores, corpus_system_pred_scores)

                    for metric, operator in zip(['MSE', 'LCC', 'SRCC'], ["<", ">", ">"]):
                        if mode == "dev":
                            value = eval(metric)
                            best_value = self.best_scores[corpus_name][metric]
                            if eval(f"{value} {operator} {best_value}"):
                                tqdm.write(f"{record_name}-{corpus_name}-{metric}={value:.4f} {operator} current best {corpus_name}-{metric}={best_value:.4f}, Saving checkpoint")
                                
                                self.best_scores[corpus_name][metric] = value
                                save_names.append(f"{mode}-{corpus_name}-{metric}-best.ckpt")

                        logger.add_scalar(
                            f"System-level-{record_name}/{corpus_name}-{mode}-{metric}",
                            eval(metric),
                            global_step=global_step,
                        )

                        # tqdm.write(f"[{record_name}] [{corpus_name}] [{mode}] System-level {metric}  = {eval(metric):.4f}")

        if mode == "dev" or mode == "test" or mode == "train_eval":
            for record_name in self.record_names:
                all_pred_score_list = []
                all_wav_name_list = []

                for corpus_name in self.datarc['corpus_names']:
                    for system_name in list(records[record_name][PRED_SCORE_IDX][corpus_name].keys()):
                        all_pred_score_list += records[record_name][PRED_SCORE_IDX][corpus_name][system_name]
                        all_wav_name_list += records[record_name][WAV_NAME_IDX][corpus_name][system_name]

                df = pd.DataFrame(list(zip(all_wav_name_list, all_pred_score_list)))
                df.to_csv(Path(self.expdir, f"{record_name}-{mode}-steps-{global_step}-answer.txt"), header=None, index=None)

        return save_names


def load_file(base_path, file):
    dataframe = pd.read_csv(Path(base_path, file), header=None)
    return dataframe
