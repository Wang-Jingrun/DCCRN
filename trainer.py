import os, sys, time
from pypesq import pesq
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.dccrn import *
from dataLoader import *
from utils import *

import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf
        self.device = torch.device(conf['device'])

        self.init_dataloader()
        self.model = DCCRN(
            rnn_hidden=self.conf['model']['rnn_hidden'], kernel_num=tuple_data(self.conf['model']['kernel_num'])
            ).to(self.device)
        self.loss_fn = loss

        optimizer = getattr(sys.modules['torch.optim'], self.conf['optimizer'])
        self.optimizer = optimizer(self.model.parameters(), lr=self.conf['learning_rate'])

        # 训练策略
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer, factor=self.conf['scheduler']['factor'],
            patience=self.conf['scheduler']['patience'], verbose=self.conf['scheduler']['verbose']
        )

        self.early_stop = EarlyStopping()


    def init_dataloader(self):
        train_dataset = SpeechDataset(os.path.join(self.conf['dataset_path'], self.conf['train_files']))
        self.train_loader = DataLoader(train_dataset, batch_size=self.conf['train']['batch_size'], shuffle=True)

        test_dataset = SpeechDataset(os.path.join(self.conf['dataset_path'], self.conf['test_files']))
        self.test_loader = DataLoader(test_dataset, batch_size=self.conf['train']['batch_size'], shuffle=True)

        num_loaders = self.conf['eval_files']
        self.eval_loaders = []
        for i in num_loaders:
            eval_dataset = SpeechDataset(os.path.join(self.conf['dataset_path'], self.conf['eval_files'][i]))
            eval_loader = DataLoader(eval_dataset, batch_size=self.conf['train']['batch_size'], shuffle=True)
            self.eval_loaders.append(eval_loader)

    def train_epoch(self, loader):
        self.model.train()
        train_ep_loss = 0.
        counter = 0
        for noisy_x, clean_x in loader:
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

        return train_ep_loss / counter

    def test_epoch(self, loader):
        self.model.eval()
        test_ep_loss = 0.
        counter = 0.
        for noisy_x, clean_x in loader:
            # get the output from the model
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)
            pred_x = self.model(noisy_x)

            # calculate loss
            val_loss = self.loss_fn(pred_x, clean_x)

            test_ep_loss += val_loss.item()
            counter += 1

        return test_ep_loss / counter

    def train(self):
        """
        To understand whether the network is being trained or not, we will output a train and test loss.
        """
        self.train_losses = []
        self.test_losses = []

        for e in range(self.conf['train']['epochs']):
            start = time.time()
            # first evaluating for comparison
            if e == 0:
                with torch.no_grad():
                    test_loss = self.test_epoch(self.test_loader)

                self.test_losses.append(test_loss)
                print("Loss before training:{:.6f}".format(test_loss))

            train_loss = self.train_epoch(self.train_loader)
            # self.scheduler.step()  # update lr
            with torch.no_grad():
                test_loss = self.test_epoch(self.test_loader)

            self.scheduler.step(test_loss)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            end = time.time()

            print("Epoch: {}/{}...".format(e + 1, self.conf['train']['epochs']),
                  "Loss: {:.6f}...".format(train_loss),
                  "Test Loss: {:.6f}...".format(test_loss),
                  "time: {:.1f}min".format((end - start) / 60))

            if self.conf['save_path']:
                self.save(e + 1)

            self.early_stop(test_loss)
            if self.early_stop.early_stop:
                print("Early stopping!")
                break

    def pesq_score(self, loader, save_sample=False):
        self.model.eval()
        test_pesq = 0.
        counter = 0.

        for noisy_x, clean_x in loader:
            # get the output from the model
            noisy_x = noisy_x.to(self.device)
            with torch.no_grad():
                pred_x = self.model(noisy_x)

            psq = 0.
            pred_x = pred_x.detach().cpu().numpy()
            for i in range(len(clean_x)):
                psq += pesq(clean_x[i], pred_x[i], 16000)

            psq /= len(clean_x)
            test_pesq += psq
            counter += 1

        if save_sample:
            soundfile.write(f'noisy{counter}.wav', noisy_x[0].cpu().detach().numpy().astype('int16'), 16000)
            soundfile.write(f'clean{counter}.wav', clean_x[0].cpu().numpy().astype('int16'), 16000)
            soundfile.write(f'enhance{counter}.wav', pred_x[0].astype('int16'), 16000)

        return test_pesq / counter

    def eval(self):
        print("\n\nModel evaluation.\n")
        start = time.time()

        counter = 0
        for loader in self.eval_loaders:
            pesq = self.pesq_score(loader)
            print("Value of PESQ in eval{}: {:.6f}".format(counter, pesq))
            si_snr = 0 - self.test_epoch(loader)
            print("Value of Si-SNR in eval{}: {:.6f}".format(counter, si_snr))
            counter += 1
            print()

        end = time.time()
        print("time: {:.1f}min".format((end - start) / 60))

    def save(self, epoch):
        os.makedirs(self.conf['model_path'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.conf['model_path'], f"model_state_{epoch}.pth"))

    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.conf['model_path'], self.conf['load_model'])))