# Airfoil Flow Condition Classification — PINN + CNN

An end-to-end AI pipeline for diagnosing aerodynamic stall on NACA airfoils. Physics-Informed Neural Networks (PINNs) generate physically constrained pressure coefficient (Cp) distributions, which are then classified by a transfer-learned ResNet18 CNN into three flow regimes: **Attached**, **Transitioning**, or **Stalled**.

Developed for **AE-301: Artificial Intelligence Using Python Lab**, NUST Department of Aerospace Engineering, Spring 2026.

**Authors:** M. Mushaf Khan, Aisha Iqbal

---

## Problem

Aerodynamic stall — flow separation caused by excessive angle of attack — is a leading cause of aviation accidents and a core safety concern in UAV design and flight control. Diagnosing flow condition in real time traditionally requires expensive CFD simulation or wind tunnel testing, both impractical for rapid design iteration or onboard diagnostics. This project replaces that cost with a lightweight, physics-grounded AI pipeline.

## Approach

**Stage 1 — PINN data generation:** A separate Physics-Informed Neural Network (4-layer MLP, Tanh activations) is trained per airfoil–angle-of-attack combination to predict the Cp(x/c) distribution along the chord. Training combines a data loss (matched against NeuralFoil-computed aerodynamic coefficients) with a physics residual loss derived from thin airfoil theory, enforcing physically realistic curve smoothness rather than relying purely on data fitting.

**Stage 2 — CNN classification:** Each PINN-generated Cp distribution is rendered as a 500×220 image and classified by a ResNet18 backbone (ImageNet pre-trained, frozen) with a fine-tuned final layer, into the three flow regimes.

**Stage 3 — Prediction tool:** `predict.py` accepts real flight parameters (airfoil, angle of attack, velocity, chord length), trains a PINN on-the-fly, and returns the predicted flow condition with a confidence score and safety status.

## Dataset

| Parameter | Value |
|---|---|
| Airfoil profiles | 11 (NACA 0006, 0009, 0012, 1408, 1412, 2412, 2415, 4412, 4415, 6412, 23012) |
| Angles of attack | 26 (−5° to +20°, 1° steps) |
| Reynolds number | 1×10⁶ |
| Total images | 286 |
| Class split | Attached 44.1% · Stalled 38.8% · Transitioning 17.1% |

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Attached | 0.83 | 1.00 | 0.91 |
| Stalled | 0.80 | 1.00 | 0.89 |
| Transitioning | 1.00 | 0.46 | 0.63 |
| **Overall accuracy** | | | **84%** |

A key finding: an earlier formula-based (non-PINN) dataset achieved a suspicious 100% test accuracy — the mathematically clean curves made classes trivially separable, meaning the CNN was memorizing the generation formula rather than learning real aerodynamic patterns. The PINN-based dataset introduces realistic physical variation, producing the lower but far more meaningful 84% accuracy. The model's weaker recall on the transitioning class (46%) reflects genuine physical ambiguity near stall onset — even experienced aerodynamicists and CFD tools struggle to draw a sharp boundary there.

## Repository Structure

```
├── pinn_generate_dataset.py    # Trains 286 PINNs, generates dataset_pinn/
├── generate_dataset.py         # Baseline formula-based dataset (for comparison)
├── train_model.py              # Trains ResNet18 classifier, saves model + metrics
├── predict.py                  # Real-time prediction tool from flight parameters
├── visualize_simulation.py     # AoA sweep visualization (attached → stalled)
├── check_dataset.py            # Random dataset sample viewer
├── dataset_pinn/                # PINN-generated Cp images (attached/stalled/transitioning)
├── dataset/                     # Formula-based baseline dataset
├── Models/                      # Trained model weights + evaluation plots
├── predictions/                 # Sample live prediction outputs
└── outputs/                     # Simulation visualizations
```

## Tech Stack

Python 3.14 · PyTorch 2.11 (CUDA 12.8) · Aerosandbox / NeuralFoil · torchvision · scikit-learn · Matplotlib

## Limitations & Future Work

Trained at a single Reynolds number (10⁶), limiting generalization to UAV-scale or transport-aircraft regimes. The Cp supervision signal comes from NeuralFoil rather than a full panel-method solver (e.g. XFOIL). Analysis is 2D only — no spanwise or 3D wing effects. Planned extensions include multi-Reynolds training, full XFOIL integration, finer transitioning sub-classes, a full Navier–Stokes PINN, and 3D wing geometry support.