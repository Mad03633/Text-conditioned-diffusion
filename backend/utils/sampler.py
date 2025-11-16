import torch
from config import T
from model.diffusion import ConditionalUNet



def load_model(model_path, device):
    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def p_sample(model, x, t, y, betas, sqrt_one_minus_ac, sqrt_recip_alphas):
    B = x.size(0)
    t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

    eps = model(x, t_batch, y)

    beta_t = betas[t]
    mean = sqrt_recip_alphas[t] * (x - beta_t / sqrt_one_minus_ac[t] * eps)

    if t > 0:
        noise = torch.randn_like(x)
        return mean + torch.sqrt(beta_t) * noise
    else:
        return mean


@torch.no_grad()
def generate_digit(model, digit, betas, sqrt_one_minus_ac, sqrt_recip_alphas, img_size=28):
    y = torch.tensor([digit], device=betas.device)
    x = torch.randn(1, 1, img_size, img_size, device=betas.device)

    for t in reversed(range(T)):
        x = p_sample(model, x, t, y, betas, sqrt_one_minus_ac, sqrt_recip_alphas)

    x = (x.clamp(-1, 1) + 1) / 2
    return x[0, 0].cpu().numpy()