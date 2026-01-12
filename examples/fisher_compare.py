import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import wandb
import math

# --- Imports from your flow_matching library ---
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Sphere, Manifold

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        return self.main(h).reshape(*sz)

class ProjectToTangent(nn.Module):
    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, t=t)

# ==========================================
# 2. DATASET & TRAINING UTILS
# ==========================================

def get_data_batch(batch_size: int = 200, device: str = "cpu"):
    """4 clusters on a 2D plane projected to sphere later"""
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = (torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2)
    x2 = x2_ + (torch.floor(x1) % 2)
    data = torch.cat([x1[:, None], x2[:, None]], dim=1).float()
    return data

def wrap_data(manifold, samples):
    center = torch.cat([torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1)
    samples = torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2
    return manifold.expmap(center, samples)

def uniform_t_infinite(batch_size, device):
    """Training always uses uniform sampling for stability"""
    while True:
        yield torch.rand(batch_size, device=device)

# ==========================================
# 3. INFERENCE SCHEDULES (The Comparison)
# ==========================================

def get_inference_time_grid(method: str, steps: int, device: str, shift: float = 1.0):
    """
    Generates the discrete time steps [t_0, t_1, ..., t_N] for the ODE solver.
    """
    if method == "linear":
        # Standard: Equal spacing
        return torch.linspace(0, 1, steps, device=device)
    
    elif method == "cosine":
        # Cosine: Small steps at start/end, big steps in middle
        # Useful if vector field is curvy near boundaries
        s = torch.linspace(0, math.pi, steps, device=device)
        return (torch.cos(s) + 1) / 2  # Maps pi->0 to 0->1 reversed, so we flip
        # Actually standard cosine map: t = (1 - cos(pi*x))/2
        x = torch.linspace(0, 1, steps, device=device)
        return 0.5 * (1 - torch.cos(x * math.pi))

    elif method == "shifted":
        # Shifted: Pushes steps towards t=1 (Data). 
        # Crucial for images/visuals where details form at the end.
        t = torch.linspace(0, 1, steps, device=device)
        # Simple power shift formula: t_new = 1 - (1-t)^shift
        # shift > 1 => spends more steps near 1
        return 1 - torch.pow(1 - t, shift)
    
    else:
        raise ValueError(f"Unknown inference method: {method}")

# ==========================================
# 4. MAIN EXPERIMENT
# ==========================================

def run_experiment(args):
    # --- Init WandB ---
    wandb.init(project="sphere-inference-comparison", config=args, name=f"infer_compare")
    config = wandb.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # --- Setup Model ---
    manifold = Sphere()
    vf = ProjectToTangent(
        MLP(input_dim=3, hidden_dim=config.hidden_dim),
        manifold=manifold,
    ).to(device)

    optim = torch.optim.Adam(vf.parameters(), lr=config.lr)
    path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

    # --- Training Loop (Standard) ---
    print(f"Starting training ({config.iterations} steps)...")
    train_gen = uniform_t_infinite(config.batch_size, device)
    
    vf.train()
    for i in range(config.iterations):
        optim.zero_grad()
        t = next(train_gen)

        x_1 = wrap_data(manifold, get_data_batch(config.batch_size, device))
        x_0 = wrap_data(manifold, torch.randn_like(get_data_batch(config.batch_size, device)))

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_v = vf(path_sample.x_t, path_sample.t)
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()

        loss.backward()
        optim.step()

        if (i + 1) % 1000 == 0:
            print(f"Iter {i+1} | Loss: {loss.item():.4f}")
            wandb.log({"train_loss": loss.item()})

    # ==========================================
    # 5. INFERENCE COMPARISON
    # ==========================================
    print("Generating comparison...")
    vf.eval()
    wrapped_vf = WrappedModel(vf)
    
    # We will run 3 solvers with different Time Grids
    methods = ["linear", "cosine", "shifted"]
    N_steps = 20 # Low steps to make differences visible
    
    # Fix the initial noise so comparisons are fair
    x_init_raw = torch.randn((2000, 2), dtype=torch.float32, device=device)
    x_init = wrap_data(manifold, x_init_raw)

    fig = plt.figure(figsize=(18, 6))
    
    for idx, method in enumerate(methods):
        # 1. Get the custom time grid
        # For 'shifted', we use shift=3.0 to exaggerate the effect
        T_grid = get_inference_time_grid(method, N_steps, device, shift=3.0 if method=="shifted" else 1.0)
        
        # 2. Run Solver
        solver = RiemannianODESolver(velocity_model=wrapped_vf, manifold=manifold)
        trajectories = solver.sample(
            x_init=x_init,
            step_size=1./N_steps, # Ignored if time_grid is provided usually, but kept for safety
            method="euler",       # Euler makes step size effects most visible
            time_grid=T_grid,
            return_intermediates=False
        )
        
        # 3. Plot
        samples = trajectories.cpu().numpy()
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        # Sphere wireframe
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color="grey", alpha=0.1)
        
        # Points
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='r', s=5, alpha=0.5)
        
        # Titles showing the time distribution
        grid_vis = T_grid.cpu().numpy()
        title_str = f"{method.upper()}\nSteps: {N_steps}"
        ax.set_title(title_str)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

        # Visualize the time steps on a small 2D bar below (optional logic, omitted for brevity)

    plt.tight_layout()
    wandb.log({"inference_comparison": wandb.Image(fig)})
    plt.close(fig)
    print("Comparison logged to WandB.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_experiment(args)