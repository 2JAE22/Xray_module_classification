import torch
import torch.nn.functional as F
def gumbel_softmax(logits, tau=1.0, hard=True):
    """
    Gumbel-Softmax 샘플링 함수
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y  # Gradient는 softmax를 따라가지만, 결과는 one-hot
    return y