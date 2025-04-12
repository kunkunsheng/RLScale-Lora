import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

from src.modules.lora_model import LoRAModule


import torch

def compute_log_probs_from_params(param_trajectories):
    """
    计算 log_probs_old 和 log_probs_new：
    - 直接基于 LoRA 参数权重计算，而不是通过 model(inputs)。
    """
    log_probs_old = []
    log_probs_new = []

    for original_param, updated_param in param_trajectories:
        # 计算原始参数的 log_probs
        log_prob_old = torch.log_softmax(original_param.flatten(), dim=0)  # 计算 log 概率
        log_probs_old.append(log_prob_old)

        # 计算更新后的参数的 log_probs
        log_prob_new = torch.log_softmax(updated_param.flatten(), dim=0)  # 计算 log 概率
        log_probs_new.append(log_prob_new)

    # 堆叠成一个张量
    log_probs_old = torch.cat(log_probs_old)
    log_probs_new = torch.cat(log_probs_new)

    return log_probs_old, log_probs_new


class PPOAgent:
    def __init__(self, model: LoRAModule, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = Adam(self.model.prepare_optimizer_params("AdjustModule"), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def compute_reward(self, generated_image, target_image):
        """ 计算 SSIM 作为奖励 """
        generated_np = generated_image.detach().cpu().numpy()
        target_np = target_image.detach().cpu().numpy()

        ssim_score = ssim(generated_np, target_np, data_range=1.0, multichannel=True)
        return ssim_score

    def update(self, old_states, actions, rewards, log_probs_old):
        """ 使用 PPO 训练 `adjust_layer` """
        new_logits = self.model.adjust_layer(old_states)
        dist = Normal(new_logits, torch.ones_like(new_logits))
        log_probs_new = dist.log_prob(actions)

        # 计算 PPO 目标
        ratios = torch.exp(log_probs_new - log_probs_old)
        advantages = rewards - rewards.mean()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()