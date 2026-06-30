import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

for label in ['attached', 'transitioning', 'stalled']:
    os.makedirs(f'dataset/{label}', exist_ok=True)

airfoils = [
    'naca0012', 'naca2412', 'naca4412', 'naca23012',
    'naca0006', 'naca0009', 'naca1408', 'naca1412',
    'naca2415', 'naca4415', 'naca6412'
]
angles = list(range(-5, 21, 1))
Re = 1e6

def get_label(cl, cd, alpha):
    if alpha > 14 or cl < 0.1:
        return 'stalled'
    elif alpha > 10 or cd > 0.02:
        return 'transitioning'
    else:
        return 'attached'

def compute_cp(x, cl, label):
    """
    Physically motivated Cp distribution.
    - Attached:      sharp suction peak near LE, smooth recovery
    - Transitioning: peak narrowing, beginning to collapse
    - Stalled:       flat, no clear peak, flow fully separated
    """
    x = np.array(x, dtype=float)
    eps = 1e-6

    if label == 'attached':
        # Strong, smooth suction peak near leading edge
        peak_loc   = 0.05
        peak_width = 0.15
        peak_mag   = 1.2 + cl * 0.8
        suction    = -peak_mag * np.exp(-((x - peak_loc) ** 2) / (2 * peak_width ** 2))
        recovery   = 0.9 * (x - peak_loc)
        cp_upper   = suction + recovery
        cp_lower   = 0.15 * x * (1 - x) + 0.05

    elif label == 'transitioning':
        # Sharper, narrower peak — starting to collapse
        peak_loc   = 0.04
        peak_width = 0.07
        peak_mag   = 1.0 + cl * 0.5
        suction    = -peak_mag * np.exp(-((x - peak_loc) ** 2) / (2 * peak_width ** 2))
        recovery   = 0.6 * (x - peak_loc) + 0.15
        cp_upper   = suction + recovery
        cp_lower   = 0.12 * x * (1 - x) + 0.05

    else:  # stalled
        # Flat, collapsed — no suction peak, separated flow
        cp_upper   = -0.3 * np.ones_like(x) + 0.1 * x
        cp_lower   = 0.1  * np.ones_like(x) - 0.05 * x

    return cp_upper, cp_lower

count = {'attached': 0, 'transitioning': 0, 'stalled': 0}

for naca in tqdm(airfoils, desc="Airfoils"):
    airfoil  = asb.Airfoil(naca)
    x        = np.array(airfoil.x())
    x_foil   = x
    y_foil   = np.array(airfoil.y())

    for alpha in angles:
        aero  = airfoil.get_aero_from_neuralfoil(alpha=float(alpha), Re=Re)
        cl    = float(np.array(aero['CL']).flatten()[0])
        cd    = float(np.array(aero['CD']).flatten()[0])
        label = get_label(cl, cd, alpha)

        cp_upper, cp_lower = compute_cp(x, cl, label)

        fig, (ax_foil, ax_cp) = plt.subplots(
            1, 2, figsize=(5.0, 2.2), dpi=100,
            gridspec_kw={'width_ratios': [1, 1.4]}
        )

        fig.suptitle(
            f'{naca}  α={alpha}°  CL={cl:.2f}  CD={cd:.4f}',
            fontsize=6.5
        )

        # ── Left: airfoil profile ─────────────────────────
        ax_foil.plot(x_foil, y_foil, color='#1a6fbd', linewidth=1.5)
        ax_foil.fill_between(x_foil, y_foil, 0, alpha=0.15, color='#1a6fbd')
        ax_foil.set_aspect('equal')
        ax_foil.axis('off')
        ax_foil.set_title('Profile', fontsize=6)

        # ── Right: Cp distribution ────────────────────────
        ax_cp.plot(x, cp_upper, color='#1a6fbd', linewidth=1.8, label='Upper')
        ax_cp.plot(x, cp_lower, color='#bd3a1a', linewidth=1.2,
                   linestyle='--', label='Lower')
        ax_cp.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax_cp.invert_yaxis()
        ax_cp.set_xlim(0, 1)
        ax_cp.set_ylim(1.0, -2.5)    # Fixed scale so CNN sees consistent axes
        ax_cp.set_xlabel('x/c', fontsize=7)
        ax_cp.set_ylabel('Cp', fontsize=7)
        ax_cp.tick_params(labelsize=6)
        ax_cp.legend(fontsize=5.5, loc='lower right')
        ax_cp.grid(True, linewidth=0.3, alpha=0.5)
        ax_cp.set_title('Cp distribution', fontsize=6)

        plt.tight_layout()
        fname = f'dataset/{label}/{naca}_aoa{alpha}.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        count[label] += 1

print("\n✅ Dataset generated!")
print(f"   Attached:      {count['attached']} images")
print(f"   Transitioning: {count['transitioning']} images")
print(f"   Stalled:       {count['stalled']} images")