import torch
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