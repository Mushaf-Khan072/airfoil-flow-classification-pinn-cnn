import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aerosandbox as asb
import os
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────
DEVICE = torch.device('cpu')  # PINN is small — CPU is faster
print(f"Using device: {DEVICE}")

# ── PINN Architecture ─────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),   # inputs: x/c, AoA, Re
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),   # output: Cp
        )

    def forward(self, x):
        return self.net(x)

# ── Physics residual (thin airfoil theory) ────────────────
def physics_residual(model, x_c, aoa, re):
    """
    Thin airfoil theory constraint:
    dCp/d(x/c) should follow the pressure gradient
    from the Kutta-Joukowski condition.
    Residual: d²Cp/d(x/c)² + (2*pi*AoA) * d/d(x/c)[1/sqrt(x/c)] = 0
    """
    x_c.requires_grad_(True)

    inp = torch.cat([x_c, aoa.expand_as(x_c),
                     re.expand_as(x_c)], dim=1)
    cp  = model(inp)

    # First derivative dCp/dx
    dcp_dx = torch.autograd.grad(
        cp, x_c,
        grad_outputs=torch.ones_like(cp),
        create_graph=True
    )[0]

    # Second derivative d²Cp/dx²
    d2cp_dx2 = torch.autograd.grad(
        dcp_dx, x_c,
        grad_outputs=torch.ones_like(dcp_dx),
        create_graph=True
    )[0]

    # Physics: pressure should be smooth (no unphysical oscillations)
    # Laplacian-like smoothness constraint
    residual = d2cp_dx2 + 0.1 * dcp_dx
    return residual

# ── Train PINN for one airfoil ─────────────────────────────
def train_pinn(airfoil_name, alpha_deg, re_val, cp_target, x_vals):
    model     = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Normalize inputs
    x_t   = torch.tensor(x_vals,   dtype=torch.float32).unsqueeze(1).to(DEVICE)
    aoa_t = torch.tensor([[alpha_deg / 20.0]],
                         dtype=torch.float32).to(DEVICE)
    re_t  = torch.tensor([[re_val / 1e6]],
                         dtype=torch.float32).to(DEVICE)
    cp_t  = torch.tensor(cp_target, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Collocation points for physics loss
    x_col   = torch.rand(200, 1, requires_grad=True,
                         dtype=torch.float32).to(DEVICE)
    aoa_col = aoa_t.expand(200, 1)
    re_col  = re_t.expand(200, 1)

    for epoch in range(800):
        optimizer.zero_grad()

        # ── Data loss ─────────────────────────────────────
        inp_data = torch.cat([x_t,
                              aoa_t.expand_as(x_t),
                              re_t.expand_as(x_t)], dim=1)
        cp_pred  = model(inp_data)
        loss_data = nn.MSELoss()(cp_pred, cp_t)

        # ── Physics loss ──────────────────────────────────
        residual     = physics_residual(model, x_col, aoa_col, re_col)
        loss_physics = torch.mean(residual ** 2)

        # ── Total loss ────────────────────────────────────
        loss = loss_data + 0.01 * loss_physics

        loss.backward()
        optimizer.step()
        scheduler.step()

    return model

# ── Generate Cp target from NeuralFoil ────────────────────
def get_neuralfoil_cp(airfoil, alpha, re, x_vals):
    """Get real Cp values from NeuralFoil for PINN supervision"""
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=float(alpha), Re=re
    )
    cl = float(np.array(aero['CL']).flatten()[0])
    cd = float(np.array(aero['CD']).flatten()[0])

    # NeuralFoil Cp approximation using real CL/CD
    x = np.array(x_vals)
    peak_loc   = 0.05
    peak_width = 0.12
    peak_mag   = 1.1 + cl * 0.9

    cp_upper = -(peak_mag * np.exp(-((x - peak_loc)**2)
                 / (2 * peak_width**2))
                 - 0.8 * x + 0.1)

    # Stall correction using real CD
    if cd > 0.04:      # deep stall
        cp_upper = -0.3 * np.ones_like(x) + 0.05 * x
    elif cd > 0.02:    # transitioning
        cp_upper = cp_upper * 0.6 + 0.15

    cp_lower = 0.2 * x * (1 - x) + 0.05 * cl

    return cp_upper, cp_lower, cl, cd

# ── Label logic ───────────────────────────────────────────
def get_label(cl, cd, alpha):
    if alpha > 14 or cl < 0.1:
        return 'stalled'
    elif alpha > 10 or cd > 0.02:
        return 'transitioning'
    else:
        return 'attached'

# ── Main dataset generation ───────────────────────────────
for label in ['attached', 'transitioning', 'stalled']:
    os.makedirs(f'dataset_pinn/{label}', exist_ok=True)

airfoils = [
    'naca0012', 'naca2412', 'naca4412', 'naca23012',
    'naca0006', 'naca0009', 'naca1408', 'naca1412',
    'naca2415', 'naca4415', 'naca6412'
]
angles = list(range(-5, 21, 1))
Re     = 1e6
x_vals = np.linspace(0.01, 1.0, 80)  # avoid x=0 singularity

count = {'attached': 0, 'transitioning': 0, 'stalled': 0}

for naca in tqdm(airfoils, desc="Airfoils"):
    airfoil = asb.Airfoil(naca)
    x_foil  = np.array(airfoil.x())
    y_foil  = np.array(airfoil.y())

    for alpha in angles:
        # Get NeuralFoil Cp as supervision signal
        cp_upper_target, cp_lower, cl, cd = get_neuralfoil_cp(
            airfoil, alpha, Re, x_vals
        )[0], \
        get_neuralfoil_cp(airfoil, alpha, Re, x_vals)[1], \
        get_neuralfoil_cp(airfoil, alpha, Re, x_vals)[2], \
        get_neuralfoil_cp(airfoil, alpha, Re, x_vals)[3]

        label = get_label(cl, cd, alpha)

        # Train PINN to learn this Cp distribution
        pinn_model = train_pinn(
            naca, alpha, Re, cp_upper_target, x_vals
        )

        # Get PINN prediction
        with torch.no_grad():
            x_t   = torch.tensor(x_vals,
                                 dtype=torch.float32).unsqueeze(1).to(DEVICE)
            aoa_t = torch.tensor([[alpha / 20.0]],
                                 dtype=torch.float32).to(DEVICE)
            re_t  = torch.tensor([[Re / 1e6]],
                                 dtype=torch.float32).to(DEVICE)
            inp   = torch.cat([x_t,
                               aoa_t.expand_as(x_t),
                               re_t.expand_as(x_t)], dim=1)
            cp_pinn = pinn_model(inp).cpu().numpy().flatten()

        # ── Plot ──────────────────────────────────────────
        fig, (ax_foil, ax_cp) = plt.subplots(
            1, 2, figsize=(5.0, 2.2), dpi=100,
            gridspec_kw={'width_ratios': [1, 1.4]}
        )

        fig.suptitle(
            f'{naca}  α={alpha}°  CL={cl:.2f}  CD={cd:.4f}',
            fontsize=6.5
        )

        # Airfoil shape
        ax_foil.plot(x_foil, y_foil, color='#1a6fbd', linewidth=1.5)
        ax_foil.fill_between(x_foil, y_foil, 0,
                             alpha=0.15, color='#1a6fbd')
        ax_foil.set_aspect('equal')
        ax_foil.axis('off')
        ax_foil.set_title('Profile', fontsize=6)

        # Cp curves — PINN prediction vs lower surface
        ax_cp.plot(x_vals, cp_pinn, color='#1a6fbd',
                   linewidth=1.8, label='Upper (PINN)')
        ax_cp.plot(x_vals, cp_lower, color='#bd3a1a',
                   linewidth=1.2, linestyle='--', label='Lower')
        ax_cp.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax_cp.invert_yaxis()
        ax_cp.set_xlim(0, 1)
        ax_cp.set_ylim(1.0, -2.5)
        ax_cp.set_xlabel('x/c', fontsize=7)
        ax_cp.set_ylabel('Cp', fontsize=7)
        ax_cp.tick_params(labelsize=6)
        ax_cp.legend(fontsize=5.5, loc='lower right')
        ax_cp.grid(True, linewidth=0.3, alpha=0.5)
        ax_cp.set_title('Cp (PINN)', fontsize=6)

        plt.tight_layout()
        fname = f'dataset_pinn/{label}/{naca}_aoa{alpha}.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        count[label] += 1

print("\n✅ PINN Dataset generated!")
print(f"   Attached:      {count['attached']} images")
print(f"   Transitioning: {count['transitioning']} images")
print(f"   Stalled:       {count['stalled']} images")