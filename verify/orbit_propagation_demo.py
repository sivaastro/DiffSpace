"""
Orbit Propagation Demo with Multiple Gravity Models and Solvers
================================================================

This script demonstrates:
1. Orbit propagation using TorchDiffEq with different gravity models
2. Comparison between gravity models (Point Mass, Nagy, Polyhedron, SPH)
3. Comparison between different ODE solvers (dopri5, dopri8, rk4)
4. Visualization of 3D orbits, XY projections, and error metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import time

# Ensure standalone_simulator is importable
ROOT = Path(__file__).resolve().parents[1].parent
sys.path.append(str(ROOT))

from standalone_simulator.models.gravity_models_fast import (
    PointMassGravity,
    NagyPolyhedronGravityFast,
    PolyhedronGravityFast,
    SphericalHarmonicsGravityFast
)
from standalone_simulator.engine.orbit_simulator_torchdiffeq import TorchDiffEqOrbitSimulator

# Constants
a, b, c = 34.4, 11.2, 11.2  # km
density = 2670e9  # kg/km^3
G = 6.674308e-20  # km^3/(kg*s^2)
volume = (4/3) * np.pi * (a/2) * (b/2) * (c/2)
mu = G * volume * density

# Paths for .mat files
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'

def create_gravity_models():
    """Create all gravity models."""
    models = {
        'Point Mass': PointMassGravity(mu),
        'Nagy': NagyPolyhedronGravityFast((a, b, c), density),
        'Polyhedron': PolyhedronGravityFast(str(MODELS_DIR / 'ParaV.mat'), str(MODELS_DIR / 'ParaF.mat')),
        'SPH': SphericalHarmonicsGravityFast(mu, str(MODELS_DIR / 'SPH.mat'), 4, (a, b, c))
    }
    return models

def run_orbit_comparison():
    """Compare orbits from different gravity models."""
    print("="*80)
    print("ORBIT COMPARISON: Multiple Gravity Models")
    print("="*80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}\n")
    
    # Initial conditions - circular orbit at 50 km
    r_orbit = 50.0
    v_circ = np.sqrt(mu / r_orbit)
    initial_state = torch.tensor(
        [r_orbit, 0.0, 0.0, 0.0, v_circ, 0.0],
        dtype=torch.float32,
        device=device
    )
    
    # Time span - 2 orbital periods
    T_orbit = 1 * np.pi * np.sqrt(r_orbit**3 / mu)
    t_span = torch.linspace(0, 2 * T_orbit, 100, device=device)
    
    print(f"Initial state: r={r_orbit} km, v={v_circ:.6f} km/s")
    print(f"Orbital period: {T_orbit:.1f} s ({T_orbit/3600:.2f} hours)")
    print(f"Simulation time: {t_span[-1].item():.1f} s ({len(t_span)} points)")
    print()
    
    # Create models and propagate
    models = create_gravity_models()
    results = {}
    
    for name, model in models.items():
        print(f"Propagating with {name}...")
        sim = TorchDiffEqOrbitSimulator(model, device=device)
        
        start = time.time()
        solution = sim.propagate(
            initial_state,
            t_span,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7,
            use_adjoint=False,
            verbose=False
        )
        elapsed = time.time() - start
        
        results[name] = {
            'solution': solution,
            'time': elapsed
        }
        print(f"  ✓ Complete in {elapsed:.2f}s")
    
    print()
    
    # Visualization
    plot_orbit_comparison(results, t_span)
    
    return results

def run_solver_comparison():
    """Compare different ODE solvers."""
    print("="*80)
    print("SOLVER COMPARISON: dopri5 vs dopri8 vs rk4")
    print("="*80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use Polyhedron model for testing
    model = PolyhedronGravityFast(str(MODELS_DIR / 'ParaV.mat'), str(MODELS_DIR / 'ParaF.mat'))
    
    # Initial conditions
    r_orbit = 50.0
    v_circ = np.sqrt(mu / r_orbit)
    initial_state = torch.tensor(
        [r_orbit, 0.0, 0.0, 0.0, v_circ, 0.0],
        dtype=torch.float32,
        device=device
    )
    
    # Time span
    T_orbit = 2 * np.pi * np.sqrt(r_orbit**3 / mu)
    t_span = torch.linspace(0, 2 * T_orbit, 1000, device=device)
    
    solvers = [
        ('dopri5', 1e-5, 1e-7, 'Adaptive RK45'),
        ('dopri8', 1e-5, 1e-7, 'Adaptive RK78'),
        ('rk4', 1e-3, 1e-5, 'Fixed RK4')
    ]
    
    results = {}
    
    for method, rtol, atol, description in solvers:
        print(f"Testing {description} ({method})...")
        sim = TorchDiffEqOrbitSimulator(model, device=device)
        
        start = time.time()
        solution = sim.propagate(
            initial_state,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol,
            use_adjoint=False,
            verbose=False
        )
        elapsed = time.time() - start
        
        results[description] = {
            'solution': solution,
            'time': elapsed,
            'method': method
        }
        print(f"  ✓ Complete in {elapsed:.2f}s")
    
    print()
    
    # Visualization
    plot_solver_comparison(results, t_span)
    
    return results

def plot_orbit_comparison(results, t_span):
    """Plot comparison of different gravity models."""
    fig = plt.figure(figsize=(16, 10))
    
    # 3D orbit plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        ax1.plot(sol[:, 0], sol[:, 1], sol[:, 2], label=name, linewidth=2, alpha=0.8)
    ax1.scatter([0], [0], [0], c='orange', s=200, marker='*', label='Asteroid')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY projection
    ax2 = fig.add_subplot(2, 3, 2)
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        ax2.plot(sol[:, 0], sol[:, 1], label=name, linewidth=2, alpha=0.8)
    ax2.scatter(0, 0, c='orange', s=200, marker='*', label='Asteroid')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Distance vs time
    ax3 = fig.add_subplot(2, 3, 3)
    t_hours = t_span.cpu().numpy() / 3600
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        dist = np.linalg.norm(sol[:, :3], axis=1)
        ax3.plot(t_hours, dist, label=name, linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Distance (km)')
    ax3.set_title('Orbital Radius vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Velocity magnitude vs time
    ax4 = fig.add_subplot(2, 3, 4)
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        vel = np.linalg.norm(sol[:, 3:], axis=1)
        ax4.plot(t_hours, vel, label=name, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Velocity (km/s)')
    ax4.set_title('Velocity Magnitude vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Computation time comparison
    ax5 = fig.add_subplot(2, 3, 5)
    names = list(results.keys())
    times = [results[name]['time'] for name in names]
    bars = ax5.bar(names, times, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    ax5.set_ylabel('Time (s)')
    ax5.set_title('Computation Time')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Trajectory divergence from Point Mass
    ax6 = fig.add_subplot(2, 3, 6)
    point_mass_sol = results['Point Mass']['solution'].cpu().numpy()
    for name, data in results.items():
        if name == 'Point Mass':
            continue
        sol = data['solution'].cpu().numpy()
        diff = np.linalg.norm(sol[:, :3] - point_mass_sol[:, :3], axis=1)
        ax6.plot(t_hours, diff, label=name, linewidth=2, alpha=0.8)
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Position Difference (km)')
    ax6.set_title('Trajectory Divergence vs Point Mass')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outfile = Path('orbit_comparison_all_models.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close()

def plot_solver_comparison(results, t_span):
    """Plot comparison of different solvers."""
    fig = plt.figure(figsize=(14, 10))
    
    # XY trajectory
    ax1 = fig.add_subplot(2, 2, 1)
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        ax1.plot(sol[:, 0], sol[:, 1], label=name, linewidth=2, alpha=0.8)
    ax1.scatter(0, 0, c='orange', s=200, marker='*')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_title('XY Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Distance vs time
    ax2 = fig.add_subplot(2, 2, 2)
    t_hours = t_span.cpu().numpy() / 3600
    for name, data in results.items():
        sol = data['solution'].cpu().numpy()
        dist = np.linalg.norm(sol[:, :3], axis=1)
        ax2.plot(t_hours, dist, label=name, linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Distance (km)')
    ax2.set_title('Orbital Radius vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Computation time
    ax3 = fig.add_subplot(2, 2, 3)
    names = list(results.keys())
    times = [results[name]['time'] for name in names]
    ax3.bar(names, times, alpha=0.7)
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Computation Time')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Trajectory difference from dopri5
    ax4 = fig.add_subplot(2, 2, 4)
    reference = list(results.values())[0]['solution'].cpu().numpy()
    for name, data in list(results.items())[1:]:
        sol = data['solution'].cpu().numpy()
        diff = np.linalg.norm(sol[:, :3] - reference[:, :3], axis=1)
        ax4.plot(t_hours, diff, label=name, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Position Difference (km)')
    ax4.set_title('Divergence from Adaptive RK45')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outfile = Path('solver_comparison.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close()

if __name__ == '__main__':
    print("="*80)
    print("STANDALONE DIFFERENTIAL SIMULATOR - ORBIT PROPAGATION DEMO")
    print("="*80)
    print()
    print(f"Asteroid: {a} × {b} × {c} km")
    print(f"μ = {mu:.6e} km³/s²")
    print()
    
    # Run comparisons
    print("Running orbit comparison with all gravity models...")
    orbit_results = run_orbit_comparison()
    
    print()
    print("Running solver comparison...")
    solver_results = run_solver_comparison()
    
    print()
    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print()
    print("Generated files:")
    print("  • orbit_comparison_all_models.png - Compare all gravity models")
    print("  • solver_comparison.png - Compare ODE solvers")
    print()
    print("Summary:")
    print(f"  • Tested {len(orbit_results)} gravity models")
    print(f"  • Tested {len(solver_results)} ODE solvers")
    print(f"  • All simulations completed successfully ✓")
