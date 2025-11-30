# Standalone Differential Simulator - Documentation

## Overview

This standalone simulator provides a complete environment for differentiable orbit propagation around small bodies (asteroids, comets) using PyTorch and TorchDiffEq.

## Architecture

### Components

```
standalone_simulator/
├── models/          # Gravity field implementations
├── engine/          # ODE integration engine
├── verify/          # Validation and comparison scripts
├── utils/           # Visualization tools
└── data/            # Data files
```

### Design Principles

1. **Differentiability**: All gravity models support PyTorch autograd
2. **Performance**: Vectorized operations (10-40x speedup over loops)
3. **Modularity**: Plug-and-play gravity models with standard interface
4. **Efficiency**: Adjoint method for O(1) memory gradient computation

## Gravity Models API

All gravity models inherit from `nn.Module` and implement:

### Required Methods

```python
class GravityModel(nn.Module):
    def compute_gravity(self, r_batch: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        Compute gravitational acceleration.
        
        Args:
            r_batch: (N, 3) position vectors [km]
            device: 'cpu' or 'cuda'
        
        Returns:
            a: (N, 3) acceleration vectors [km/s²]
        """
        pass
    
    def compute_potential(self, r_batch: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        Compute gravitational potential.
        
        Args:
            r_batch: (N, 3) position vectors [km]
            device: 'cpu' or 'cuda'
        
        Returns:
            U: (N,) potential values [km²/s²]
        """
        pass
```

### Available Models

#### 1. PointMassGravity
Simple Keplerian gravity field.
```python
model = PointMassGravity(mu)  # mu in km³/s²
```

#### 2. NagyPolyhedronGravityFast
Analytical cuboid gravity (Werner & Scheeres formulation).
```python
model = NagyPolyhedronGravityFast(
    dims=(a, b, c),     # km
    density=2670e9      # kg/km³
)
```

#### 3. PolyhedronGravityFast
Face/vertex polyhedron gravity.
```python
model = PolyhedronGravityFast(
    vertex_file='ParaV.mat',
    face_file='ParaF.mat'
)
```

#### 4. SphericalHarmonicsGravityFast
Spherical harmonics expansion (up to degree 4).
```python
model = SphericalHarmonicsGravityFast(
    mu=mu,
    coeffs_file='SPH.mat',
    max_degree=4,
    dims=(a, b, c)
)
```

## TorchDiffEq Integration

### TorchDiffEqOrbitSimulator

```python
sim = TorchDiffEqOrbitSimulator(gravity_model, device='cpu')

solution = sim.propagate(
    initial_state,      # (6,) [x,y,z,vx,vy,vz]
    t_span,             # (n,) evaluation times
    method='dopri5',    # ODE solver
    rtol=1e-7,          # Relative tolerance
    atol=1e-9,          # Absolute tolerance
    use_adjoint=True,   # Enable adjoint method
    verbose=True        # Print info
)
# Returns: (n, 6) state trajectory
```

### Solver Methods

| Method | Type | Order | Recommended For |
|--------|------|-------|-----------------|
| `dopri5` | Adaptive | 5 | General use, default |
| `dopri8` | Adaptive | 8 | High accuracy |
| `adaptive_heun` | Adaptive | 2 | Fast, low accuracy |
| `rk4` | Fixed | 4 | Debugging, comparison |
| `euler` | Fixed | 1 | Educational only |

### Adjoint Method

For optimization problems, use `use_adjoint=True`:

**Benefits:**
- Memory: O(1) instead of O(n×states)
- Speed: Comparable to forward pass
- Accuracy: Controlled by `rtol` and `atol`

**When to use:**
- Parameter estimation
- Trajectory optimization
- Large time spans
- Many evaluation points

## Optimization Patterns

### Parameter Estimation

```python
# Make model parameters optimizable
model.density.requires_grad_(True)

optimizer = torch.optim.Adam([model.density], lr=1e-3)

for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward propagation with adjoint
    solution = sim.propagate(
        initial_state, t_span,
        use_adjoint=True
    )
    
    # Loss vs observations
    loss = (solution - observations).pow(2).sum()
    
    # Backward (uses adjoint!)
    loss.backward()
    
    optimizer.step()
```

### Trajectory Optimization

```python
# Optimize initial velocity
v0 = initial_state[3:].clone().requires_grad_(True)

for i in range(iterations):
    state = torch.cat([initial_state[:3], v0])
    
    solution = sim.propagate(state, t_span, use_adjoint=True)
    
    # Target final position
    loss = (solution[-1, :3] - target).pow(2).sum()
    loss.backward()
    
    v0.data -= learning_rate * v0.grad
    v0.grad.zero_()
```

## Visualization

### Available Plotting Functions

```python
from standalone_simulator.utils.plotting import (
    plot_orbit_3d,
    plot_orbit_xy,
    plot_orbit_metrics
)

# 3D trajectory
plot_orbit_3d(solution, title='3D Orbit', outfile='orbit.png')

# XY projection
plot_orbit_xy(solution, title='Orbit (XY)', outfile='xy.png')

# Distance and velocity over time
plot_orbit_metrics(solution, t_span, title='Metrics', outfile='metrics.png')
```

## Performance Optimization

### Batching

Process multiple positions simultaneously:
```python
# Single position
r = torch.tensor([[50.0, 0.0, 0.0]])
a = model.compute_gravity(r)  # (1, 3)

# Batched (much faster!)
r_batch = torch.tensor([
    [50.0, 0.0, 0.0],
    [45.0, 5.0, 0.0],
    # ... more positions
])
a_batch = model.compute_gravity(r_batch)  # (N, 3)
```

### GPU Acceleration

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = NagyPolyhedronGravityFast(dims, density).to(device)
sim = TorchDiffEqOrbitSimulator(model, device=device)

initial_state = initial_state.to(device)
t_span = t_span.to(device)

solution = sim.propagate(initial_state, t_span)
```

### Tolerance Tuning

Balance accuracy vs. speed:

```python
# High accuracy (slow)
solution = sim.propagate(..., rtol=1e-9, atol=1e-11)

# Medium accuracy (default)
solution = sim.propagate(..., rtol=1e-7, atol=1e-9)

# Low accuracy (fast)
solution = sim.propagate(..., rtol=1e-5, atol=1e-7)
```

## Extending the Simulator

### Adding New Gravity Models

1. Inherit from `nn.Module`
2. Implement `compute_gravity(r_batch, device)`
3. Implement `compute_potential(r_batch, device)`
4. Ensure differentiability (use PyTorch ops)

```python
class CustomGravity(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = nn.Parameter(torch.tensor(params))
    
    def compute_gravity(self, r_batch, device='cpu'):
        # Your implementation
        # Must return (N, 3) accelerations
        pass
    
    def compute_potential(self, r_batch, device='cpu'):
        # Your implementation
        # Must return (N,) potentials
        pass
```

### Custom ODE Functions

For complex dynamics beyond gravity:

```python
from standalone_simulator.engine.orbit_simulator_torchdiffeq import ODEFunc

class CustomODEFunc(nn.Module):
    def __init__(self, gravity_model):
        super().__init__()
        self.gravity = gravity_model
    
    def forward(self, t, state):
        r = state[:3]
        v = state[3:]
        
        # Gravity
        a_grav = self.gravity.compute_gravity(r.unsqueeze(0)).squeeze(0)
        
        # Additional forces (e.g., solar radiation pressure)
        a_srp = self.compute_srp(r, t)
        
        a_total = a_grav + a_srp
        
        return torch.cat([v, a_total])
```

## Validation Scripts

### compare_gravity_fields_absolute_error.py

Generates 2D heatmaps of absolute error between two gravity models.

**Usage:**
```powershell
python verify\compare_gravity_fields_absolute_error.py `
    --modelA nagy --modelB poly `
    --xmin -50 --xmax 50 --nx 200 `
    --ymin -50 --ymax 50 --ny 200
```

**Output:** `abs_error_fields.png`

### orbit_propagation_demo.py

Comprehensive orbit comparison demo:
- Multiple gravity models
- Multiple ODE solvers
- Energy conservation analysis
- 3D visualization

**Usage:**
```powershell
python verify\orbit_propagation_demo.py
```

**Outputs:**
- Orbit comparison plots
- Solver comparison plots
- Energy drift metrics
- Performance benchmarks

## Troubleshooting

### Common Issues

**1. Singularity errors near origin**
- Solution: Use `differentiable_check()` to add epsilon to near-zero values
- Already implemented in all models

**2. NaN gradients**
- Check initial conditions (avoid starting at origin)
- Reduce tolerance (`rtol`, `atol`)
- Use `create_graph=False` in `torch.autograd.grad()`

**3. Slow performance**
- Enable GPU: `device='cuda'`
- Use batched operations
- Reduce tolerance for adaptive solvers
- Use `dopri5` instead of `dopri8`

**4. Memory errors**
- Use `use_adjoint=True` for long integrations
- Reduce number of evaluation points
- Process in batches

## References

- **TorchDiffEq**: [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- **Polyhedron Gravity**: Werner & Scheeres (1997), "Exterior gravitation of a polyhedron"
- **Adjoint Method**: Chen et al. (2018), "Neural Ordinary Differential Equations"

## Build More

- Add new gravity models by implementing the standard interface
- Integrate with parameter estimation pipelines
- Extend to multi-body systems
- Add perturbation forces (SRP, drag, third-body)
- Implement event detection (surface impact, escape)
