import os
import math
import torch
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import KeywordSpottingTrainDataset, KeywordSpottingEvalDataset, load_data


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.
                
                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        language = self.datarc['language']
        rnd = np.random.RandomState(seed=self.datarc['seed'])
        sr = self.datarc['sampling_rate']

        train_csv = self.datarc['train_csv']
        test_csv = self.datarc['test_csv']
        dev_csv = self.datarc['dev_csv']
        print('Using train csv: ', train_csv)
        x_train, y_train = load_data(train_csv, rnd, sr)
        x_dev, y_dev = load_data(dev_csv, rnd, sr)
        x_test, y_test = load_data(test_csv, rnd, sr)

        classes = np.unique(y_train)

        if language == 'lithuanian':
            noise_csv = self.datarc['noise_csv']
            bg_audio = load_data(noise_csv, rnd, sr)[0]
            bg_audio = [noise * rnd.random() * 0.1 for noise in bg_audio]
            bg_audio = np.array(bg_audio)
            rnd.shuffle(bg_audio)
            x_test = np.concatenate([x_test, bg_audio[:10]])
            x_dev = np.concatenate([x_dev, bg_audio[10:20]])
            y_test.extend(["silence"]*10)
            y_dev.extend(["silence"]*10)
            classes = np.append(classes, ["silence"])

        cls2label = {label: i for i, label in enumerate(classes.tolist())}
        self.lbl2class = {v: k for k, v in cls2label.items()}
        num_classes = len(classes)
        shuffle = self.datarc['shuffle']
        self.batch_size = self.datarc['batch_size']
        if language == 'lithuanian':
            # For lithuanian there is special training data handling
            self.train_dataset = KeywordSpottingTrainDataset(
                           datas=x_train,
                           labels=y_train,
                           classes=cls2label,
                           epochs=self.datarc['steps_in_epoch'],
                           rnd=rnd,
                           bg_audio=bg_audio[20:],
                           batch_size=self.batch_size,
                           shuffle=shuffle)
        else:
            self.train_dataset = KeywordSpottingEvalDataset(
                                                  x_train,
                                                  y_train,
                                                  classes=cls2label)
        self.dev_dataset = KeywordSpottingEvalDataset(x_dev,
                                                  y_dev,
                                                  classes=cls2label)
        self.test_dataset = KeywordSpottingEvalDataset(x_test,
                                                   y_test,
                                                   classes=cls2label)
        self.language = language
        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        self.model = Model(
            output_class_num=num_classes,
            **self.modelrc
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            split: string
                'train'
                    will always be called before the training loop

                'dev', 'test', or more
                    defined by the 'eval_dataloaders' field in your downstream config
                    these will be called before the evaluation loops during the training loop

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if split == 'train':
            return self._get_train_dataloader(self.train_dataset, epoch)
        elif split == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == 'test':
            return self._get_eval_dataloader(self.test_dataset)


    def _get_train_dataloader(self, dataset, epoch: int):
        dl = DataLoader(
            dataset, batch_size=self.datarc['batch_size'],
            shuffle=False,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )
        if self.language == 'lithuanian':
            self.train_dataset.prepare_train_data()
        return dl


    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    # Interface
    def forward(self, split, features, labels, records, **kwargs):
        """
        Args:
            split: string
                'train'
                    when the forward is inside the training loop

                'dev', 'test' or more
                    when the forward is inside the evaluation loop

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        predicted = self.model(features)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices

        # records["filename"] += filenames
        records['loss'].append(loss.item())
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        records["predict"] += [self.lbl2class[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.lbl2class[idx] for idx in labels.cpu().tolist()]

        return loss


    # interface
    # def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
    #     """
    #     Args:
    #         split: string
    #             'train':
    #                 records and batchids contain contents for `log_step` batches
    #                 `log_step` is defined in your downstream config
    #                 eg. downstream/example/config.yaml

    #             'dev', 'test' or more:
    #                 records and batchids contain contents for the entire evaluation dataset

    #         records:
    #             defaultdict(list), contents already prepared by self.forward

    #         logger:
    #             Tensorboard SummaryWriter
    #             please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
    #             preventing conflict with the logging of other tasks

    #         global_step:
    #             The global_step when training, which is helpful for Tensorboard logging

    #         batch_ids:
    #             The batches contained in records when enumerating over the dataloader

    #         total_batch_num:
    #             The total amount of batches in the dataloader
    #     
    #     Return:
    #         a list of string
    #             Each string is a filename we wish to use to save the current model
    #             according to the evaluation result, like the best.ckpt on the dev set
    #             You can return nothing or an empty list when no need to save the checkpoint
    #     """
    #     save_names = []
    #     for key, values in records.items():
    #         average = torch.FloatTensor(values).mean().item()
    #         logger.add_scalar(
    #             f'example/{split}-{key}',
    #             average,
    #             global_step=global_step
    #         )
    #         if split == 'dev' and key == 'acc' and average > self.best_score:
    #             self.best_score = torch.ones(1) * average
    #             save_names.append(f'{split}-best.ckpt')
    #     return save_names


    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["loss", "acc"]:
            values = records[key]
            average = sum(values) / len(values)
            logger.add_scalar(
                f'KWS/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(self.expdir + '/' + "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        with open(self.expdir + '/' + f"{mode}_predict.txt", "w") as file:
            lines = [f"{i}\n" for i in records["predict"]]
            file.writelines(lines)

        with open(self.expdir + '/' + f"{mode}_truth.txt", "w") as file:
            lines = [f"{i}\n" for i in records["truth"]]
            file.writelines(lines)

        return save_names
