# Standalone Orbit Simulator

A self-contained PyTorch-based orbit propagation simulator with differentiable gravity models and TorchDiffEq integration.

## Features

- **Multiple Gravity Models**: Point Mass, Nagy (cuboid), Polyhedron, Spherical Harmonics
- **Adaptive ODE Solvers**: dopri5 (RK45), dopri8 (RK78), rk4, adaptive_heun
- **Adjoint Method**: O(1) memory gradient computation for optimization
- **Batched Operations**: Vectorized gravity field computations (10-40x faster)
- **Visualization**: 3D orbit plots, XY projections, and error metrics

## Project Structure

```
standalone_simulator/
├── models/              # Gravity model implementations
│   ├── gravity_models_fast.py     # Optimized gravity models
│   ├── Transformation_Matrix.py   # Polyhedron geometry utilities
│   ├── ParaV.mat                  # Polyhedron vertices data
│   ├── ParaF.mat                  # Polyhedron faces data
│   └── SPH.mat                    # Spherical harmonics coefficients
├── engine/              # Orbit propagation engine
│   └── orbit_simulator_torchdiffeq.py  # TorchDiffEq-based simulator
├── verify/              # Comparison and validation scripts
│   ├── compare_gravity_fields_absolute_error.py  # Compare gravity models
│   └── orbit_propagation_demo.py                 # Full orbit comparison demo
├── utils/               # Visualization utilities
│   └── plotting.py      # 3D plots, XY projections, metrics
└── data/                # Data files documentation
    └── README.txt       # Info about required .mat files
```

## Quick Start

### Installation

```powershell
conda activate pytor
python -m pip install torch torchdiffeq matplotlib numpy scipy
```

### Compare Gravity Fields

Generate absolute error plots between two gravity models:

```powershell
python standalone_simulator\verify\compare_gravity_fields_absolute_error.py --modelA nagy --modelB poly --nx 200 --ny 200
```

Available models: `point`, `nagy`, `poly`, `sph`

### Run Orbit Propagation Demo

Compare orbits using different gravity models and solvers:

```powershell
python standalone_simulator\verify\orbit_propagation_demo.py
```

This generates:
- 3D orbit visualizations
- Gravity model comparisons
- ODE solver comparisons
- Energy conservation metrics

## Usage Examples

### Basic Orbit Propagation

```python
import torch
from standalone_simulator.models.gravity_models_fast import NagyPolyhedronGravityFast
from standalone_simulator.engine.orbit_simulator_torchdiffeq import TorchDiffEqOrbitSimulator

# Create gravity model
model = NagyPolyhedronGravityFast((34.4, 11.2, 11.2), density=2670e9)

# Initialize simulator
sim = TorchDiffEqOrbitSimulator(model, device='cpu')

# Initial state [x, y, z, vx, vy, vz]
initial_state = torch.tensor([50.0, 0.0, 0.0, 0.0, 0.001, 0.0])

# Time span
t_span = torch.linspace(0, 3600, 100)  # 1 hour, 100 points

# Propagate orbit
solution = sim.propagate(initial_state, t_span, method='dopri5')
```

### Gradient-Based Optimization

```python
# Make initial velocity optimizable
v0 = initial_state[3:].clone().requires_grad_(True)

# Use adjoint method for O(1) memory gradients
solution = sim.propagate(
    torch.cat([initial_state[:3], v0]),
    t_span,
    use_adjoint=True  # Enables efficient backpropagation
)

# Compute loss and backpropagate
loss = (solution[-1, :3] - target_position).pow(2).sum()
loss.backward()  # Gradients computed via adjoint method
```

## Gravity Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **PointMass** | Simple μ/r² gravity | Fast, low-fidelity |
| **Nagy** | Analytical cuboid field | Medium fidelity, analytical |
| **Polyhedron** | Face/vertex mesh | High fidelity, irregular shapes |
| **SPH** | Spherical harmonics (degree 4) | High fidelity, smooth fields |

## ODE Solvers

- `dopri5` - Adaptive RK45 (Dormand-Prince), recommended default
- `dopri8` - Adaptive RK78, higher accuracy
- `adaptive_heun` - Adaptive RK23
- `rk4` - Fixed-step RK4
- `euler` - Fixed-step Euler

## Performance

- **10-40x faster** than loop-based implementations via batched autograd
- Supports GPU acceleration (CUDA)
- Adjoint method reduces memory usage from O(n) to O(1) for gradients

## Notes

- All gravity models implement `compute_gravity()` and `compute_potential()` methods
- Compatible with PyTorch autograd for end-to-end differentiation
- .mat files (ParaV.mat, ParaF.mat, SPH.mat) must be present in `models/` directory
