import numpy as np
import os, torch, random, soundfile, pyroomacoustics
from numpy.linalg import norm
from torch.utils.data import Dataset, DataLoader


def load_wav(path, wav_file):
    wav, sr = soundfile.read(os.path.join(path, wav_file))
    wav = wav.astype('float32')
    # 能量归一化
    wav = wav / ((np.sqrt(np.sum(wav ** 2)) / ((wav.size) + 1e-7)) + 1e-7)
    return wav


class SpeechDataset(Dataset):
    def __init__(self, dataset_path, clean_files, noise_files, rir_files, wav_dur=3, is_train=True):
        super(SpeechDataset, self).__init__()
        clean_files = np.loadtxt(clean_files, dtype='str').tolist()
        train_files_num = int(len(clean_files) * 0.9)
        if is_train:
            self.clean_files = clean_files[:train_files_num]
        else:
            self.clean_files = clean_files[train_files_num:]

        self.noise_files = np.loadtxt(noise_files, dtype='str').tolist()
        self.rir_files = np.loadtxt(rir_files, dtype='str').tolist()
        self.dataset_path = dataset_path
        self.max_len = wav_dur * 16000

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        # 读取干净语音
        clean_wav = load_wav(self.dataset_path, self.clean_files[index])
        # 裁剪至固定长度
        clean_wav = self.cut(clean_wav)
        # plot_stft_spec(clean_wav, 16000, n_fft=512, win_l=400, hop_l=200, picname='./output/stft-clean')
        # soundfile.write(f'./output/CLEAN.wav', clean_wav.astype('int16'), 16000)

        # 加混响
        rir_wav = self.add_rir(clean_wav)
        # plot_stft_spec(rir_wav, 16000, n_fft=512, win_l=400, hop_l=200, picname='./output/stft-rir')
        # soundfile.write(f'./output/RIR.wav', rir_wav.astype('int16'), 16000)

        # 加噪声
        noisy_wav = self.add_noise(rir_wav)
        # plot_stft_spec(noisy_wav, 16000, n_fft=512, win_l=400, hop_l=200, picname='./output/stft-noisy')
        # soundfile.write(f'./output/NOISY.wav', noisy_wav.astype('int16'), 16000)
        return torch.from_numpy(noisy_wav).type(torch.FloatTensor), \
               torch.from_numpy(clean_wav).type(torch.FloatTensor)

    def cut(self, wav):
        # 用0填充，保持每个batch中的长度一致
        if wav.shape[0] <= self.max_len:
            shortage = self.max_len - wav.shape[0]
            wav = np.pad(wav, (0, shortage), 'constant')  # 用0填充

        start = np.int64(random.random() * (wav.shape[0] - self.max_len))
        return wav[start: start + self.max_len]

    def add_rir(self, wav):
        rir_file = random.choice(self.rir_files)
        rir, _ = soundfile.read(os.path.join(self.dataset_path, rir_file))

        out_wav = np.convolve(wav, rir)
        return out_wav[:wav.shape[0]]

    def add_noise(self, wav):
        noise_file = random.choice(self.noise_files)
        noise = load_wav(self.dataset_path, noise_file)

        # 对噪声进行裁剪
        len_speech = wav.shape[0]
        len_noise = noise.shape[0]
        # 噪声文件前面一段时间可能没有声音
        start = random.randint(1000, len_noise - len_speech)
        noise = noise[start: start + len_speech]

        snr = random.randint(-5, 20)
        add_nosie = noise / norm(noise) * norm(wav) / (10.0 ** (0.05 * snr))
        return wav + add_nosie


class EvalDataset(Dataset):
    def __init__(self, data_paths):
        super(EvalDataset, self).__init__()
        datas = np.loadtxt(data_paths, dtype='str')
        self.noisy_files = datas[:, 0].tolist()
        self.clean_files = datas[:, 1].tolist()

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        noisy_wav, _ = load_wav(self.noisy_files[index])
        clean_wav, _ = load_wav(self.clean_files[index])
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)
