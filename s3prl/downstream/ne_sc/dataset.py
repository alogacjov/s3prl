# import random
# 
import torch
# import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import pandas as pd
import soundfile as sound
from torch.utils.data.dataset import Dataset

# SAMPLE_RATE = 16000
# EXAMPLE_WAV_MIN_SEC = 5
# EXAMPLE_WAV_MAX_SEC = 20
# EXAMPLE_DATASET_SIZE = 200
# 
# 
# class RandomDataset(Dataset):
#     def __init__(self, **kwargs):
#         self.class_num = 48
# 
#     def __getitem__(self, idx):
#         samples = random.randint(EXAMPLE_WAV_MIN_SEC * SAMPLE_RATE, EXAMPLE_WAV_MAX_SEC * SAMPLE_RATE)
#         wav = torch.randn(samples)
#         label = random.randint(0, self.class_num - 1)
#         return wav, label
# 
#     def __len__(self):
#         return EXAMPLE_DATASET_SIZE
# 
#     def collate_fn(self, samples):
#         wavs, labels = [], []
#         for wav, label in samples:
#             wavs.append(wav)
#             labels.append(label)
#         return wavs, labels


def load_data(data_csv, rnd, sr, language=None):
    '''Loads the given wav files
    
    Parameters
    ----------
    data_csv: str
        path of metadata for all files to load
    rnd: np.random.RandomState
    sr: int
        sampling rate

    Returns
    -------
    : (np.array, list)
        x and y

    '''
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels = data_df['label'].to_list()

    x, y = list(), list()
    for wav, label in zip(wavpath, labels):
        stereo, fs = sound.read(wav)
        if language is not None and language == 'mandarin':
            stereo_trim, index = librosa.effects.trim(stereo, top_db=20)
        else:
            stereo = stereo / np.abs(stereo).max()
        if fs != sr:
            stereo = librosa.resample(stereo, fs, sr)
        if stereo.shape[0] > sr:
            if language is not None and language == 'mandarin':
                start = (stereo.shape[0] - sr) // 2
            else:
                start = rnd.choice(len(stereo) - sr + 1)
            x.append(stereo[start:start+sr])
        else:
            x.append(np.pad(stereo, (0, sr-stereo.shape[0])))
        
        y.append(label)
    return np.array(x), y


class KeywordSpottingEvalDataset(Dataset):
    def __init__(self, datas, labels, classes, **kwargs):
        self.datas = datas
        self.labels = labels
        self.classes = classes

    def __getitem__(self, idx):
        x, y = self.datas[idx], self.labels[idx]
        y = self.classes[y]
        return x, y

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)

class KeywordSpottingTrainDataset(Dataset):
    def __init__(self, datas, labels,
                 classes, epochs, rnd=None, bg_audio=None,
                 batch_size=32, shuffle=True, language=None, **kwargs):
        self.unknowns = list()
        self.commands = list()
        self._split_unknown(datas, labels)

        self.datas = list()

        self.bg_audio = bg_audio
        self.classes = classes
        self.add_noise = "silence" in self.classes
        
        self.rnd = rnd

        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.language = language
        self.prepare_train_data()

    def _split_unknown(self, datas, labels):
        '''Split unknown labels from command labels'''
        for wav, label in zip(datas, labels):
            if label == "unknown":
                self.unknowns.append((wav, label))
            else:
                self.commands.append((wav, label))

    def prepare_train_data(self):
        '''Pick and shuffle from unknowns, commands, bg_audio

        Size of data is batch_size*7 (why?)

        '''
        if self.language != 'lithuanian':
            return
        self.datas = list()
        for _ in range(self.batch_size*self.epochs):
            # random float between [0.0,1.0) of uniform
            coin = self.rnd.random()
            if coin < 0.1:
                # 10% of cases, unknown added to current data
                unk = self.rnd.choice(len(self.unknowns))
                unknown = self.unknowns[unk]
                self.datas.append(unknown)
            elif coin < 0.15:
                # 5% of cases, silence added to current data
                sil = self.rnd.choice(len(self.bg_audio))
                silence = self.bg_audio[sil]
                self.datas.append((silence, "silence"))
            else:
                # 85% of cases, command added to current data
                com = self.rnd.choice(len(self.commands))
                command = self.commands[com]
                self.datas.append(command)
        # Shuffle if necessary     
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True:
            self.rnd.shuffle(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        datas = self.datas[index]
        x, y = datas
        y = self.classes[y]
        return x, y

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)

    def __data_generation(self, batch):
        X, y = list(), list()
        for wav, label in batch:
            X.append(wav)
            y.append(self.classes[label])

        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y
