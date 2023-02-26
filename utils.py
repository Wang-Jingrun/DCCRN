import torch, os
import numpy as np
from pypesq import pesq


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)


def loss(inputs, label):
    return -(si_snr(inputs, label))


def pesq_metric(y_hat, bd, sample_rate):
    # PESQ
    with torch.no_grad():
        y_hat = y_hat.cpu().numpy()
        y = bd['y'].cpu().numpy()  # target signal

        sum = 0
        for i in range(len(y)):
            sum += pesq(y[i, 0], y_hat[i, 0], sample_rate)

        sum /= len(y)
        return torch.tensor(sum)


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


class EarlyStopping:
    def __init__(self,patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        os.makedirs(path, exist_ok=True)
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'model_state_best.pth'))
        self.val_loss_min = val_loss


def tuple_data(data):
    """
    将yaml读取到的字符数据转换成元组
    """
    return eval(repr(data).replace('\'', ''))