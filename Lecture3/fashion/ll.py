import numpy as np
import torch
import torch.distributions as dist
import sys

def compute_log_likelihood(model, test_loader, num_samples=5000):
    model.eval()
    total_ll = 0.0
    num_data = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for x, _ in test_loader:  # 假设test_loader返回（数据,标签）
            x = x.to(device)
            batch_size = x.size(0)
            
            # 编码器输出变分后验的参数
            mu, logvar = model.encode(x)
            q_z_given_x = dist.Normal(mu, torch.exp(0.5 * logvar))
            
            # 从q(z|x)中采样z
            z = q_z_given_x.rsample((num_samples,))  # 形状: (S, batch_size, latent_dim)
            
            # 计算生成概率p(x|z)
            recon_x = []
            for i in range(num_samples):
                recon_x.append(model.decode(z[i]))  # 假设decoder返回伯努利分布的logits
            recon_x = torch.stack(recon_x)
            x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
            p_x_given_z = dist.Bernoulli(logits=recon_x).log_prob(x_expanded)#.sum(dim=[-1, -2, -3])  # 形状: (S, batch_size)
            print(p_x_given_z.shape)
            sys.exit()
            
            # 计算先验p(z)（标准正态）
            p_z = dist.Normal(0, 1).log_prob(z).sum(dim=-1)  # 形状: (S, batch_size)
            
            # 计算变分后验q(z|x)
            q_z_given_x_logprob = q_z_given_x.log_prob(z).sum(dim=-1)  # 形状: (S, batch_size)
            
            # 重要性权重
            log_weights = p_x_given_z + p_z - q_z_given_x_logprob  # 形状: (S, batch_size)
            
            # 对数似然估计（log-mean-exp技巧，避免数值不稳定）
            log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
            total_ll += log_likelihood.sum().item()
            num_data += batch_size

    avg_log_likelihood = total_ll / num_data
    return avg_log_likelihood
