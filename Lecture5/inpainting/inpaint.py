import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
image_size = 64
timesteps = 1000
epochs = 50
lr = 1e-4

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CelebA dataset
dataset = CelebA(root='./data', split='all', download=True, transform=transform)
print("The length of dataset is:", len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Diffusion model scheduler
class DiffusionScheduler:
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        
    def add_noise(self, x_0, t):
        """Forward diffusion process: add noise to the input image x_0 at timestep t."""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
        
    def sample_prev_timestep(self, x_t, model_pred, t):
        """Sample x_{t-1} from x_t using model prediction."""
        #self.alphas = self.alphas.to(device)
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
            
        # Compute x_{t-1}
        pred_noise, pred_x0 = model_pred
        x_t_minus_one = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise
        
        return x_t_minus_one

# U-Net model definition
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256, 512], time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial convolution
        self.conv0 = nn.Conv2d(in_channels * 2, hidden_dims[0], 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList([
            Block(hidden_dims[i], hidden_dims[i+1], time_emb_dim) 
            for i in range(len(hidden_dims)-1)
        ])
        
        # Upsampling path
        self.ups = nn.ModuleList([
            Block(hidden_dims[i], hidden_dims[i-1], time_emb_dim, up=True) 
            for i in range(len(hidden_dims)-1, 0, -1)
        ])
        
        self.output = nn.Conv2d(hidden_dims[0], out_channels, 1)
        
    def forward(self, x, t, mask):
        # Concatenate image and mask as input
        x = torch.cat([x * (1 - mask), mask], dim=1)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, 32)
        t_emb = self.time_mlp(t_emb)
        
        # Initial convolution
        x = self.conv0(x)
        
        # Downsampling path
        residual_inputs = []
        for down in self.downs:
            x = down(x, t_emb)
            residual_inputs.append(x)
        
        # Upsampling path
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat([x, residual_x], dim=1)
            x = up(x, t_emb)
        
        return self.output(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """Generate sinusoidal positional embeddings"""
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = emb.to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# Initialize model and scheduler
model = SimpleUnet().to(device)
scheduler = DiffusionScheduler(timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    pbar = tqdm(dataloader)
    for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Create random mask (1 indicates missing region)
        mask = torch.zeros_like(images)
        for j in range(images.size(0)):
            # Random rectangular mask
            h_start = np.random.randint(0, image_size - 20)
            w_start = np.random.randint(0, image_size - 20)
            h_end = h_start + np.random.randint(10, 30)
            w_end = w_start + np.random.randint(10, 30)
            mask[j, :, h_start:h_end, w_start:w_end] = 1
        
        # Random timestep
        t = torch.randint(0, timesteps, (images.size(0),)).to(device)
        
        # Forward diffusion process
        noisy_images, noise = scheduler.add_noise(images, t)
        
        # Mask out parts of the image using the mask
        masked_images = noisy_images * (1 - mask)
        
        # Model prediction
        pred_noise = model(masked_images, t, mask)
        
        # Compute loss - only over the masked region
        loss = F.mse_loss(pred_noise * mask, noise * mask)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

# Inpainting function
def inpaint(image, mask, model, scheduler, steps=timesteps):
    """Perform image inpainting using the trained diffusion model."""
    model.eval()
    with torch.no_grad():
        # Initialize as random noise
        x_t = torch.randn_like(image).to(device)
        
        # Denoise step by step
        for t in range(steps-1, -1, -1):
            # Create tensor for current timestep
            t_batch = torch.full((image.size(0),), t, device=device, dtype=torch.long)
            
            # Model prediction
            pred_noise = model(x_t, t_batch, mask)
            
            # Sample previous timestep
            x_t = scheduler.sample_prev_timestep(x_t, (pred_noise, None), t_batch)
            
            # Keep known regions
            if t > 0:
                # Add noise to the known regions of the original image
                _, known_region_noisy = scheduler.add_noise(image, t_batch)
                x_t = x_t * mask + known_region_noisy * (1 - mask)
            else:
                # Final step: replace known regions with the original image
                x_t = x_t * mask + image * (1 - mask)
                
        return x_t

# Test inpainting
def test_inpainting():
    # Load a test image
    test_image, _ = next(iter(dataloader))
    test_image = test_image[:1].to(device)
    
    # Create center rectangular mask
    mask = torch.zeros_like(test_image)
    h_start, w_start = image_size//4, image_size//4
    h_end, w_end = 3*image_size//4, 3*image_size//4
    mask[:, :, h_start:h_end, w_start:w_end] = 1
    
    # Apply mask
    masked_image = test_image * (1 - mask)
    
    # Perform inpainting
    restored = inpaint(masked_image, mask, model, scheduler)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(test_image[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(masked_image[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.title("Masked")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(restored[0].cpu().detach().permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.title("Restored")
    plt.axis('off')
    
    plt.savefig('inpainting_result.png')
    plt.show()

# Test after training completed
test_inpainting()

# Save model
torch.save(model.state_dict(), 'diffusion_inpainting.pth')

