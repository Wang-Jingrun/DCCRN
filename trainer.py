import os, sys, time
# from pypesq import pesq
from pesq import pesq
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.dccrn import *
from dataLoader import *

from utils.early_stopping import *
from utils.loss import *
from utils.utils import *

import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        self.init_dataloader()
        self.model = DCCRN(
            rnn_hidden=self.config['model']['rnn_hidden'], kernel_num=tuple_data(self.config['model']['kernel_num'])
            ).to(self.device)
        self.loss_fn = loss

        optimizer = getattr(sys.modules['torch.optim'], self.config['optimizer'])
        self.optimizer = optimizer(self.model.parameters(), lr=self.config['learning_rate'])

        # 训练策略
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer, factor=self.config['scheduler']['factor'],
            patience=self.config['scheduler']['patience'], verbose=self.config['scheduler']['verbose']
        )

        self.early_stop = EarlyStopping(verbose=True)

    def init_dataloader(self):
        if self.config['VCTK_dataset']:
            dataset_config = self.config['VCTK_dataset']
            train_dataset = VCTKTrain(dataset_config['path'], dataset_config['trainset'],
                                      wav_dur=dataset_config['wav_dur'], is_trian=True)
            self.train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'],
                                           num_workers=self.config['train']['num_workers'], shuffle=True)

            validate_dataset = VCTKTrain(dataset_config['path'], dataset_config['trainset'],
                                      wav_dur=dataset_config['wav_dur'], is_trian=False)
            self.validate_loader = DataLoader(validate_dataset, batch_size=self.config['train']['batch_size'],
                                              num_workers=self.config['train']['num_workers'])

            self.eval_loaders = []
            eval_dataset = VCTKEval(dataset_config['path'], dataset_config['testset'])
            eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=self.config['train']['num_workers'])
            self.eval_loaders.append(eval_loader)

        elif self.config['DeepXi_dataset']:
            dataset_config = self.config['DeepXi_dataset']
            train_dataset = DeepXiTrain(dataset_config['path'], dataset_config['train_clean_files'],
                                               dataset_config['train_noise_files'], wav_dur=dataset_config['wav_dur'])
            self.train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'],
                                           num_workers=self.config['train']['num_workers'], shuffle=True)

            validate_dataset = DeepXiVal(dataset_config['path'], dataset_config['validate_files'],
                                                wav_dur=dataset_config['wav_dur'])
            self.validate_loader = DataLoader(validate_dataset, batch_size=self.config['train']['batch_size'],
                                              num_workers=self.config['train']['num_workers'])

            snrs = tuple_data(dataset_config['test_snrs'])
            self.eval_loaders = []
            for snr in snrs:
                eval_dataset = DeepXiEval(dataset_config['path'], dataset_config['test_files'], snr)
                eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=self.config['train']['num_workers'])
                self.eval_loaders.append(eval_loader)

    def train_epoch(self):
        self.model.train()
        train_ep_loss = 0.
        counter = 0
        for noisy_x, clean_x in self.train_loader:
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

            # zero  gradients
            self.model.zero_grad()

            # get the output from the model
            pred_x = self.model(noisy_x)

            # calculate loss
            loss = self.loss_fn(pred_x, clean_x)
            loss.backward()
            self.optimizer.step()

            train_ep_loss += loss.item()
            counter += 1

        clear_cache()
        return train_ep_loss / counter

    def test_epoch(self):
        self.model.eval()
        val_ep_loss = 0.
        counter = 0.
        for noisy_x, clean_x in self.validate_loader:
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

            # get the output from the model
            pred_x = self.model(noisy_x)

            # calculate loss
            val_loss = self.loss_fn(pred_x, clean_x)

            val_ep_loss += val_loss.item()
            counter += 1

        clear_cache()
        return val_ep_loss / counter

    def train(self):
        """
        To understand whether the network is being trained or not, we will output a train and test loss.
        """
        self.train_losses = []
        self.test_losses = []

        for e in range(self.config['train']['epochs']):
            start = time.time()

            train_loss = self.train_epoch()
            with torch.no_grad():
                test_loss = self.test_epoch()

            self.scheduler.step(test_loss)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            end = time.time()

            print("Epoch: {}/{}...".format(e + 1, self.config['train']['epochs']),
                  "Loss: {:.6f}...".format(train_loss),
                  "Test Loss: {:.6f}...".format(test_loss),
                  "time: {:.1f}min".format((end - start) / 60))

            self.save('model_state_last.pth')
            self.early_stop(test_loss, self.model, self.config['model_path'])
            if self.early_stop.early_stop:
                print("Early stopping!")
                break

    def eval(self, save_sample=False):
        self.model.eval()
        print("\n\nModel evaluation.\n")

        si_snrs = []
        pesqs = []
        for index, loader in enumerate(self.eval_loaders):
            pesq_total, si_snr_total, counter = 0., 0., 0.

            start = time.time()
            for noisy_x, clean_x in loader:
                noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)
                with torch.no_grad():
                    pred_x = self.model(noisy_x)
                    clean_x = clean_x[:, :pred_x.shape[1]]

                # 计算si_snr
                si_snr_ = si_snr(pred_x, clean_x)
                si_snr_total += si_snr_.item()

                clean_x = clean_x.detach().cpu().numpy()
                noisy_x = noisy_x.detach().cpu().numpy()
                pred_x = pred_x.detach().cpu().numpy()

                # 计算pesq
                psq = 0.
                for i in range(len(clean_x)):
                    # psq += pesq(clean_x[i], pred_x[i], 16000) # MOS-PESQ
                    psq += pesq(16000, clean_x[i], pred_x[i], 'wb')  # WB-PESQ 宽带
                    # psq += pesq(16000, clean_x[i], pred_x[i], 'nb')  # NB-PESQ 窄带
                psq /= len(clean_x)
                pesq_total += psq

                counter += 1

            si_snrs.append(si_snr_total / counter)
            pesqs.append(pesq_total / counter)
            end = time.time()

            print("Dataset[{}]...".format(index),
                  "Si-SNR: {:.6f}...".format(si_snrs[index]),
                  "PESQ: {:.6f}...".format(pesqs[index]),
                  "time: {:.1f}min".format((end - start) / 60))

            if save_sample:
                clean_x = clean_x[0] / ((np.sqrt(np.sum(clean_x[0] ** 2)) / (clean_x.shape[1] + 1e-7)) + 1e-7)
                noisy_x = noisy_x[0] / ((np.sqrt(np.sum(noisy_x[0] ** 2)) / (noisy_x.shape[1] + 1e-7)) + 1e-7)
                pred_x = pred_x[0] / ((np.sqrt(np.sum(pred_x[0] ** 2)) / (pred_x.shape[1] + 1e-7)) + 1e-7)
                soundfile.write(f'./output/clean{counter}_testset{index}.wav', clean_x.astype('int16'), 16000)
                soundfile.write(f'./output/noisy{counter}_testset{index}.wav', noisy_x.astype('int16'), 16000)
                soundfile.write(f'./output/enhance{counter}_testset{index}.wav', pred_x.astype('int16'), 16000)

        print("Average...",
              "Si-SNR: {:.6f}...".format(sum(si_snrs) / len(self.eval_loaders)),
              "PESQ: {:.6f}...".format(sum(pesqs) / len(self.eval_loaders)))

    def save(self, pth_name='model_state.pth'):
        os.makedirs(self.config['model_path'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['model_path'], pth_name))

    def load(self, model_state):
        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'], model_state)))