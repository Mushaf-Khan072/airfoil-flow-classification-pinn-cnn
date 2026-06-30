import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

DEVICE = torch.device('cpu')

# ── PINN ──────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

def physics_residual(model, x_c, aoa, re):
    x_c.requires_grad_(True)
    inp      = torch.cat([x_c, aoa.expand_as(x_c),
                          re.expand_as(x_c)], dim=1)
    cp       = model(inp)
    dcp_dx   = torch.autograd.grad(
        cp, x_c, grad_outputs=torch.ones_like(cp),
        create_graph=True)[0]
    d2cp_dx2 = torch.autograd.grad(
        dcp_dx, x_c, grad_outputs=torch.ones_like(dcp_dx),
        create_graph=True)[0]
    return d2cp_dx2 + 0.1 * dcp_dx

def get_cp_target(airfoil, alpha, re, x_vals):
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=float(alpha), Re=re)
    cl   = float(np.array(aero['CL']).flatten()[0])
    cd   = float(np.array(aero['CD']).flatten()[0])
    x    = np.array(x_vals)
    pl, pw = 0.05, 0.12
    pm     = 1.1 + cl * 0.9
    cp_u   = -(pm * np.exp(-((x-pl)**2)/(2*pw**2)) - 0.8*x + 0.1)
    if cd > 0.04:
        cp_u = -0.3 * np.ones_like(x) + 0.05 * x
    elif cd > 0.02:
        cp_u = cp_u * 0.6 + 0.15
    cp_l = 0.2 * x * (1-x) + 0.05 * cl
    return cp_u, cp_l, cl, cd

def train_pinn(alpha, re, cp_target, x_vals):
    model     = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.995)
    x_t   = torch.tensor(x_vals,
                         dtype=torch.float32).unsqueeze(1)
    aoa_t = torch.tensor([[alpha/20.0]], dtype=torch.float32)
    re_t  = torch.tensor([[re/1e6]],    dtype=torch.float32)
    cp_t  = torch.tensor(cp_target,
                         dtype=torch.float32).unsqueeze(1)
    x_col   = torch.rand(200, 1, requires_grad=True,
                         dtype=torch.float32)
    aoa_col = aoa_t.expand(200, 1)
    re_col  = re_t.expand(200, 1)

    for epoch in range(800):
        optimizer.zero_grad()
        inp      = torch.cat([x_t, aoa_t.expand_as(x_t),
                              re_t.expand_as(x_t)], dim=1)
        cp_pred  = model(inp)
        loss_d   = nn.MSELoss()(cp_pred, cp_t)
        residual = physics_residual(model, x_col, aoa_col, re_col)
        loss_p   = torch.mean(residual**2)
        loss     = loss_d + 0.01 * loss_p
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model

# ── Config ────────────────────────────────────────────────
NACA    = 'naca2412'
ALPHAS  = [2, 6, 10, 14, 18]
RE      = 1e6
x_vals  = np.linspace(0.01, 1.0, 80)

# Class color + label mapping
def get_label_color(cd, alpha):
    if alpha > 14 or True and cd > 0.04:
        return 'stalled',      '#F85149'
    elif alpha > 10 or cd > 0.02:
        return 'transitioning','#F0883E'
    else:
        return 'attached',     '#3FB950'

airfoil = asb.Airfoil(NACA)
x_foil  = np.array(airfoil.x())
y_foil  = np.array(airfoil.y())

# ── Figure setup ──────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor='#0D1117')
fig.suptitle(
    f'Flow Condition Simulation — {NACA.upper()}  ·  Re = {RE:.0e}  ·  AoA Sweep',
    fontsize=15, fontweight='bold', color='#E6EDF3', y=0.98
)

print(f"Simulating {NACA.upper()} across {len(ALPHAS)} angles of attack...")

results = []
for i, alpha in enumerate(ALPHAS):
    print(f"  Training PINN for α={alpha}°...", end='', flush=True)
    cp_u_t, cp_l, cl, cd = get_cp_target(airfoil, alpha, RE, x_vals)
    pinn = train_pinn(alpha, RE, cp_u_t, x_vals)

    with torch.no_grad():
        x_t   = torch.tensor(x_vals,
                              dtype=torch.float32).unsqueeze(1)
        aoa_t = torch.tensor([[alpha/20.0]], dtype=torch.float32)
        re_t  = torch.tensor([[RE/1e6]],    dtype=torch.float32)
        inp   = torch.cat([x_t, aoa_t.expand_as(x_t),
                           re_t.expand_as(x_t)], dim=1)
        cp_pinn = pinn(inp).numpy().flatten()

    label, color = get_label_color(cd, alpha)
    results.append({
        'alpha': alpha, 'cl': cl, 'cd': cd,
        'cp_pinn': cp_pinn, 'cp_lower': cp_l,
        'label': label, 'color': color
    })
    print(f" done! → {label.upper()}  CL={cl:.3f}")

# ── Plot each AoA ─────────────────────────────────────────
for i, r in enumerate(results):
    # Top row: airfoil profiles
    ax_foil = fig.add_subplot(2, 5, i + 1)
    ax_foil.set_facecolor('#161B22')
    ax_foil.plot(x_foil, y_foil, color=r['color'], linewidth=2)
    ax_foil.fill_between(x_foil, y_foil, 0,
                         alpha=0.3, color=r['color'])
    ax_foil.set_aspect('equal')
    ax_foil.axis('off')
    ax_foil.set_title(
        f"α = {r['alpha']}°\n{r['label'].upper()}",
        fontsize=11, fontweight='bold',
        color=r['color'], pad=6
    )
    ax_foil.text(
        0.5, -0.18,
        f"CL={r['cl']:.3f}  CD={r['cd']:.4f}",
        transform=ax_foil.transAxes,
        fontsize=8.5, color='#8B949E',
        ha='center'
    )

    # Bottom row: Cp curves
    ax_cp = fig.add_subplot(2, 5, i + 6)
    ax_cp.set_facecolor('#161B22')
    ax_cp.spines[:].set_color('#30363D')
    ax_cp.tick_params(colors='#8B949E', labelsize=7)

    ax_cp.plot(x_vals, r['cp_pinn'], color=r['color'],
               linewidth=2.0, label='Upper (PINN)')
    ax_cp.plot(x_vals, r['cp_lower'], color='#8B949E',
               linewidth=1.2, linestyle='--', label='Lower')
    ax_cp.axhline(0, color='#30363D', linewidth=0.8)
    ax_cp.invert_yaxis()
    ax_cp.set_xlim(0, 1)
    ax_cp.set_ylim(1.0, -2.5)
    ax_cp.set_xlabel('x/c', fontsize=8, color='#8B949E')
    ax_cp.set_ylabel('Cp', fontsize=8, color='#8B949E')
    ax_cp.grid(True, linewidth=0.3, alpha=0.4, color='#30363D')
    if i == 0:
        ax_cp.legend(fontsize=7, loc='lower right',
                     facecolor='#161B22', edgecolor='#30363D',
                     labelcolor='#8B949E')

# ── Flow progression arrow ────────────────────────────────
fig.text(0.5, 0.505,
         '── Increasing Angle of Attack ──▶  Attached → Transitioning → Stalled',
         ha='center', fontsize=10, color='#58A6FF',
         fontstyle='italic')

plt.tight_layout(rect=[0, 0, 1, 0.96])

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/flow_simulation.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()

print("\n✅ Simulation saved to outputs/flow_simulation.png")