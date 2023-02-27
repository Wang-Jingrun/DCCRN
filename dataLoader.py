import torch
import random
import soundfile
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SpeechDataset(Dataset):
    def __init__(self, data_paths, wav_dur=3, sr=16000):
        super(SpeechDataset, self).__init__()
        datas = np.loadtxt(data_paths, dtype='str')
        self.clean_files = datas[:, 1].tolist()
        self.noisy_files = datas[:, 0].tolist()
        self.max_len = wav_dur * sr

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        #读取含噪语音
        noisy_wav, _ = soundfile.read(self.noisy_files[index], dtype='int16')
        noisy_wav = noisy_wav.astype('float32')

        # 读取干净语音
        clean_wav, _ = soundfile.read(self.clean_files[index], dtype='int16')
        clean_wav = clean_wav.astype('float32')

        # 裁剪语音至相同长度
        noisy_wav, clean_wav = self.cut(noisy_wav, clean_wav)
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)

    def cut(self, noisy, clean):
        # 用0填充，保持每个batch中的长度一致
        wav_len = len(noisy)
        if wav_len <= self.max_len:
            shortage = self.max_len - wav_len
            noisy = np.pad(noisy, (0, shortage), 'constant')  # 用0填充
            clean = np.pad(clean, (0, shortage), 'constant')  # 用0填充
            wav_len = self.max_len

        # start = 0
        start = random.randint(0, (wav_len - self.max_len))
        return noisy[start: start + self.max_len], clean[start: start + self.max_len]

