import os, sys, time
from pypesq import pesq
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.dccrn import *
from dataLoader import *
from utils import *

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
        train_dataset = SpeechDataset(os.path.join(self.config['dataset_path'], self.config['train_files']))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'],
                                       num_workers=self.config['train']['num_workers'], shuffle=True)

        test_dataset = SpeechDataset(os.path.join(self.config['dataset_path'], self.config['test_files']))
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['train']['batch_size'],
                                      num_workers=self.config['train']['num_workers'], shuffle=True)

        num_loaders = self.config['eval_files']
        self.eval_loaders = []
        for i in num_loaders:
            eval_dataset = SpeechDataset(os.path.join(self.config['dataset_path'], self.config['eval_files'][i]))
            eval_loader = DataLoader(eval_dataset, batch_size=self.config['train']['batch_size'],
                                     num_workers=self.config['train']['num_workers'], shuffle=True)
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

        for e in range(self.config['train']['epochs']):
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

            print("Epoch: {}/{}...".format(e + 1, self.config['train']['epochs']),
                  "Loss: {:.6f}...".format(train_loss),
                  "Test Loss: {:.6f}...".format(test_loss),
                  "time: {:.1f}min".format((end - start) / 60))

            self.save('model_state_last.pth')
            self.early_stop(test_loss, self.model, self.config['model_path'])
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

    def save(self, pth_name='model_state.pth'):
        os.makedirs(self.config['model_path'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['model_path'], pth_name))

    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'], self.config['load_model'])))