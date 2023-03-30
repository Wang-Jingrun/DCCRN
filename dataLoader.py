import numpy as np
import os, torch, random, soundfile, math
from numpy.linalg import norm
from torch.utils.data import Dataset, DataLoader


def load_wav(path, wav_file):
    wav, sr = soundfile.read(os.path.join(path, wav_file))
    wav = wav.astype('float32')
    # 能量归一化
    # wav = wav / ((np.sqrt(np.sum(wav ** 2)) / ((wav.size) + 1e-7)) + 1e-7)
    return wav


def add_noise(wav, noise, snr):
    if noise.shape[0] < wav.shape[0]:
        short_num = math.ceil(wav.shape[0] / noise.shape[0])
        noise = np.tile(noise, short_num)

    # 对噪声进行裁剪
    len_speech = wav.shape[0]
    len_noise = noise.shape[0]
    # 噪声文件前面一段时间可能没有声音
    start = random.randint(0, len_noise - len_speech)
    noise = noise[start: start + len_speech]

    add_nosie = noise / (norm(noise) + 1e-7) * norm(wav) / (10.0 ** (0.05 * snr))
    return wav + add_nosie


class VCTKTrain(Dataset):
    def __init__(self, dataset_path, train_files, wav_dur=3, is_trian=True):
        super(VCTKTrain, self).__init__()
        train_files = np.loadtxt(os.path.join(dataset_path, train_files), dtype='str').tolist()
        if is_trian:
            self.train_files = train_files[:10700]
        else:
            self.train_files = train_files[10700:]

        self.dataset_path = dataset_path
        self.max_len = wav_dur * 16000

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        # 读取干净语音
        clean_wav = load_wav(os.path.join(self.dataset_path, 'clean_trainset_28spk_wav'),
                             self.train_files[index] + '.wav')
        noisy_wav = load_wav(os.path.join(self.dataset_path, 'noisy_trainset_28spk_wav'),
                             self.train_files[index] + '.wav')

        # 裁剪至固定长度
        clean_wav, noisy_wav = self.cut(clean_wav, noisy_wav)
        # soundfile.write(f'./output/CLEAN.wav', clean_wav.astype('int16'), 16000)
        # soundfile.write(f'./output/NOISY.wav', noisy_wav.astype('int16'), 16000)
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)

    def cut(self, clean_wav, noisy_wav):
        # 用0填充，保持每个batch中的长度一致
        if clean_wav.shape[0] <= self.max_len:
            shortage = self.max_len - clean_wav.shape[0]
            clean_wav = np.pad(clean_wav, (0, shortage), 'constant')  # 用0填充
            noisy_wav = np.pad(noisy_wav, (0, shortage), 'constant')  # 用0填充

        start = np.int64(random.random() * (clean_wav.shape[0] - self.max_len))
        return clean_wav[start: start + self.max_len], noisy_wav[start: start + self.max_len]


class VCTKEval(Dataset):
    def __init__(self, dataset_path, test_files):
        super(VCTKEval, self).__init__()
        self.test_files = np.loadtxt(os.path.join(dataset_path, test_files), dtype='str')
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        clean_wav = load_wav(os.path.join(self.dataset_path, 'clean_testset_wav'),
                             self.test_files[index] + '.wav')
        noisy_wav = load_wav(os.path.join(self.dataset_path, 'noisy_testset_wav'),
                             self.test_files[index] + '.wav')
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)


class DeepXiTrain(Dataset):
    def __init__(self, dataset_path, clean_files, noise_files, wav_dur=3):
        super(DeepXiTrain, self).__init__()
        self.clean_files = np.loadtxt(os.path.join(dataset_path, clean_files), dtype='str').tolist()
        self.noise_files = np.loadtxt(os.path.join(dataset_path, noise_files), dtype='str').tolist()
        self.dataset_path = dataset_path
        self.max_len = wav_dur * 16000

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        # 读取干净语音
        clean_wav = load_wav(self.dataset_path, self.clean_files[index])

        # 裁剪至固定长度
        clean_wav = self.cut(clean_wav)
        # soundfile.write(f'./output/CLEAN.wav', clean_wav.astype('int16'), 16000)

        # 加混响
        # rir_wav = self.add_rir(clean_wav)
        # soundfile.write(f'./output/RIR.wav', rir_wav.astype('int16'), 16000)

        # 加噪声
        noise_file = random.choice(self.noise_files)
        noise_wav = load_wav(self.dataset_path, noise_file)
        snr = random.randint(-10, 20)
        noisy_wav = add_noise(clean_wav, noise_wav, snr)
        # soundfile.write(f'./output/NOISY.wav', noisy_wav.astype('int16'), 16000)
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)

    def cut(self, wav):
        # 用0填充，保持每个batch中的长度一致
        if wav.shape[0] <= self.max_len:
            shortage = self.max_len - wav.shape[0]
            wav = np.pad(wav, (0, shortage), 'constant')  # 用0填充

        start = np.int64(random.random() * (wav.shape[0] - self.max_len))
        return wav[start: start + self.max_len]

    # def add_rir(self, wav):
    #     rir_file = random.choice(self.rir_files)
    #     rir, _ = soundfile.read(os.path.join(self.dataset_path, rir_file))
    #
    #     out_wav = np.convolve(wav, rir)
    #     return out_wav[:wav.shape[0]]


class DeepXiVal(Dataset):
    def __init__(self, dataset_path, val_files, wav_dur=3):
        super(DeepXiVal, self).__init__()
        self.val_files = np.loadtxt(os.path.join(dataset_path, val_files), dtype='str').tolist()
        self.dataset_path = dataset_path
        self.max_len = wav_dur * 16000

    def __len__(self):
        return len(self.val_files)

    def __getitem__(self, index):
        clean_wav = load_wav(os.path.join(self.dataset_path, 'val_clean_speech'),
                             self.val_files[index] + '.wav')
        noise_wav = load_wav(os.path.join(self.dataset_path, 'val_noise'),
                             self.val_files[index] + '.wav')

        # 裁剪至固定长度
        clean_wav = self.cut(clean_wav)

        # 加噪声
        snr = int(self.val_files[index].split('_')[-1].replace('dB', ''))
        noisy_wav = add_noise(clean_wav, noise_wav, snr)
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)

    def cut(self, wav):
        # 用0填充，保持每个batch中的长度一致
        if wav.shape[0] <= self.max_len:
            shortage = self.max_len - wav.shape[0]
            wav = np.pad(wav, (0, shortage), 'constant')  # 用0填充

        start = np.int64(random.random() * (wav.shape[0] - self.max_len))
        return wav[start: start + self.max_len]


class DeepXiEval(Dataset):
    def __init__(self, dataset_path, test_files, snr):
        super(DeepXiEval, self).__init__()
        self.test_files = np.loadtxt(os.path.join(dataset_path, test_files), dtype='str')
        self.dataset_path = dataset_path
        self.snr = snr

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        clean_wav = load_wav(os.path.join(self.dataset_path, 'test_clean_speech'),
                             self.test_files[index] + '.wav')
        noisy_wav = load_wav(os.path.join(self.dataset_path, 'test_noisy_speech'),
                             self.test_files[index] + f'_{self.snr}dB.wav')
        return torch.from_numpy(noisy_wav), torch.from_numpy(clean_wav)
