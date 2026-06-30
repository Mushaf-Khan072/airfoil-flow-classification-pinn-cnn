import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import aerosandbox as asb
import io
import os

# ── Constants ─────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES      = ['attached', 'stalled', 'transitioning']
NU_AIR       = 1.5e-5   # kinematic viscosity of air (m²/s)

# ── Load trained CNN ──────────────────────────────────────
def load_model():
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(
        'models/airfoil_pinn_cnn.pth',
        map_location=DEVICE
    ))
    model.eval()
    return model.to(DEVICE)

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

def train_pinn_for_prediction(alpha, re, cp_target, x_vals):
    model     = PINN().to('cpu')   # CPU faster for small PINN
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

    print("   Training PINN", end='', flush=True)
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
        if (epoch+1) % 200 == 0:
            print('.', end='', flush=True)
    print(' done!')
    return model

# ── Generate image ─────────────────────────────────────────
def generate_image(naca, alpha, re, velocity, chord):
    airfoil = asb.Airfoil(naca)
    x_foil  = np.array(airfoil.x())
    y_foil  = np.array(airfoil.y())
    x_vals  = np.linspace(0.01, 1.0, 80)

    cp_upper_t, cp_lower, cl, cd = get_cp_target(
        airfoil, alpha, re, x_vals)

    pinn_model = train_pinn_for_prediction(
        alpha, re, cp_upper_t, x_vals)

    with torch.no_grad():
        x_t   = torch.tensor(x_vals,
                              dtype=torch.float32).unsqueeze(1)
        aoa_t = torch.tensor([[alpha/20.0]], dtype=torch.float32)
        re_t  = torch.tensor([[re/1e6]],    dtype=torch.float32)
        inp   = torch.cat([x_t, aoa_t.expand_as(x_t),
                           re_t.expand_as(x_t)], dim=1)
        cp_pinn = pinn_model(inp).numpy().flatten()

    fig, (ax_foil, ax_cp) = plt.subplots(
        1, 2, figsize=(5.0, 2.2), dpi=100,
        gridspec_kw={'width_ratios': [1, 1.4]}
    )
    fig.suptitle(
        f'{naca}  α={alpha}°  V={velocity}m/s  '
        f'c={chord}m  Re={re:.2e}',
        fontsize=6.5
    )
    ax_foil.plot(x_foil, y_foil, color='#1a6fbd', linewidth=1.5)
    ax_foil.fill_between(x_foil, y_foil, 0,
                         alpha=0.15, color='#1a6fbd')
    ax_foil.set_aspect('equal')
    ax_foil.axis('off')
    ax_foil.set_title('Profile', fontsize=6)

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

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0.05)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert('RGB'), cl, cd

# ── Predict ───────────────────────────────────────────────
def predict(naca, alpha, re, velocity, chord):
    print(f"\n🔍 Predicting flow condition...")

    cnn_model     = load_model()
    image, cl, cd = generate_image(naca, alpha, re,
                                   velocity, chord)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs    = cnn_model(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        pred_idx   = probs.argmax().item()
        pred_label = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # ── Physical interpretation ───────────────────────────
    interpretations = {
        'attached':      '✅ Safe — flow fully attached, '
                         'lift generation normal',
        'transitioning': '⚠️  Caution — flow separating, '
                         'approaching stall',
        'stalled':       '❌ Danger — flow separated, '
                         'significant lift loss'
    }

    print(f"\n{'='*50}")
    print(f"  Airfoil      : {naca.upper()}")
    print(f"  AoA          : {alpha}°")
    print(f"  Velocity     : {velocity} m/s")
    print(f"  Chord        : {chord} m")
    print(f"  Reynolds No. : {re:.3e}")
    print(f"  CL           : {cl:.4f}")
    print(f"  CD           : {cd:.4f}")
    print(f"  L/D Ratio    : {cl/cd:.1f}")
    print(f"{'='*50}")
    print(f"  PREDICTION   : {pred_label.upper()}")
    print(f"  Confidence   : {confidence:.1f}%")
    print(f"  Status       : {interpretations[pred_label]}")
    print(f"{'='*50}")
    print("\n  Class probabilities:")
    for i, cls in enumerate(CLASSES):
        bar = '█' * int(probs[i].item() * 30)
        print(f"  {cls:>15}: {probs[i].item()*100:5.1f}%  {bar}")

    os.makedirs('predictions', exist_ok=True)
    fname = f'predictions/{naca}_aoa{alpha}_V{velocity}.png'
    image.save(fname)
    print(f"\n✅ Image saved to {fname}")

# ── Input interface ───────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("   Airfoil Flow Condition Predictor")
    print("   PINN + CNN | Aerospace AI Tool")
    print("=" * 50)

    print("\nAvailable airfoils:")
    print("  naca0006, naca0009, naca0012")
    print("  naca1408, naca1412")
    print("  naca2412, naca2415")
    print("  naca4412, naca4415")
    print("  naca6412, naca23012")

    naca  = input("\nEnter airfoil name: ").strip().lower()
    alpha = float(input("Enter angle of attack (°): ").strip())

    print("\nReynolds number input mode:")
    print("  1. Enter velocity + chord (recommended)")
    print("  2. Enter Reynolds number directly")
    mode  = input("Choose (1 or 2): ").strip()

    if mode == '1':
        velocity = float(input("Enter freestream velocity (m/s): ").strip())
        chord    = float(input("Enter chord length (m): ").strip())
        re       = (velocity * chord) / NU_AIR
        print(f"  → Computed Re = {re:.3e}")
    else:
        re       = float(input("Enter Reynolds number: ").strip())
        velocity = re * NU_AIR
        chord    = 1.0
        print(f"  → Using Re = {re:.3e}")

    predict(naca, alpha, re, velocity, chord)