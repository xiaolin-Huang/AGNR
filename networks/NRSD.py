import torch
import torch.nn as nn
import torch.nn.functional as F

class AddNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma=sigma

    def forward(self, x):
        gaussian_noise = torch.randn(x.size()) * self.sigma
        noisy_features = x + gaussian_noise.cuda()
        # Randomly mask noise features
        mask = torch.rand(noisy_features.size()) > 0.5
        masked_noisy_features = noisy_features.clone()
        masked_noisy_features[mask] = 0
        return masked_noisy_features


class DistillationLoss(nn.Module):
    def __init__(self, alfa=0.5, beta=1.0):
        super(DistillationLoss, self).__init__()
        self.alfa = alfa
        self.beta = beta

    def forward(self, teacher_logits, student_logits):
        # Compute mse between features
        reconstruction_loss = F.mse_loss(teacher_logits, student_logits, reduction='mean')

        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                            F.softmax(teacher_logits, dim=-1)+1e-10, reduction='mean')
        # Minimize the sum of the two mutual information
        loss = self.alfa*reconstruction_loss + self.beta * kl_loss

        return loss