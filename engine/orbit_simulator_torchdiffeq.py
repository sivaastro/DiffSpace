# Copied from workspace root orbit_simulator_torchdiffeq.py for standalone use
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Original content below
"""
Orbit simulator using torchdiffeq for adjoint-based gradient computation.
Combines adaptive stepping with O(1) memory gradients.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint, odeint_adjoint
from typing import Tuple, Optional
import time

class ODEFunc(nn.Module):
	"""
	Wrapper for dynamics function to work with odeint_adjoint.
	Must inherit from nn.Module for adjoint method to work.
	"""
    
	def __init__(self, gravity_model, device='cpu'):
		super().__init__()
		self.gravity_model = gravity_model
		self.device = device
    
	def forward(self, t, state):
		"""
		ODE right-hand side: dy/dt = f(y, t)
        
		Args:
			t: Current time (scalar)
			state: (6,) or (batch, 6) state vector
            
		Returns:
			state_dot: Time derivative
		"""
		# Handle both single state and batched states
		if state.dim() == 1:
			r = state[:3]
			v = state[3:]
            
			r_batch = r.unsqueeze(0)
			a = self.gravity_model.compute_gravity(r_batch, device=str(self.device))
			a = a.squeeze(0).to(self.device)
            
			return torch.cat([v, a])
		else:
			r = state[:, :3]
			v = state[:, 3:]
            
			a = self.gravity_model.compute_gravity(r, device=str(self.device))
			a = a.to(self.device)
            
			return torch.cat([v, a], dim=1)


class TorchDiffEqOrbitSimulator:
	"""
	Orbit simulator using torchdiffeq for efficient differentiable integration.
    
	Features:
	- Adaptive stepping (dopri5, dopri8, etc.)
	- Adjoint method for O(1) memory gradients
	- Production-ready for optimization and parameter estimation
	"""
    
	def __init__(self, gravity_model, device='cpu'):
		"""
		Initialize simulator.
        
		Args:
			gravity_model: Gravity model with compute_gravity() method
			device: 'cpu' or 'cuda'
		"""
		self.gravity_model = gravity_model
		self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
		self.ode_func = ODEFunc(gravity_model, str(self.device))
    
	def propagate(self, 
				  initial_state: torch.Tensor,
				  t_span: torch.Tensor,
				  method: str = 'dopri5',
				  rtol: float = 1e-7,
				  atol: float = 1e-9,
				  use_adjoint: bool = True,
				  verbose: bool = True) -> torch.Tensor:
		"""
		Propagate orbit using torchdiffeq.
        
		Args:
			initial_state: (6,) initial state [x, y, z, vx, vy, vz]
			t_span: (n,) tensor of evaluation times
			method: Integration method
				   'dopri5' - Adaptive RK45 (Dormand-Prince, default)
				   'dopri8' - Adaptive RK78 (higher accuracy)
				   'adaptive_heun' - Adaptive RK23
				   'rk4' - Fixed-step RK4
				   'euler' - Fixed-step Euler
			rtol: Relative tolerance for adaptive methods
			atol: Absolute tolerance for adaptive methods
			use_adjoint: If True, use adjoint method for backprop (O(1) memory)
			verbose: Print information
            
		Returns:
			solution: (n, 6) tensor of states at evaluation times
		"""
		initial_state = initial_state.to(self.device)
		t_span = t_span.to(self.device)
        
		if verbose:
			print(f"{'='*80}")
			print(f"TorchDiffEq Orbit Propagation")
			print(f"{'='*80}")
			print(f"Method: {method}")
			print(f"Tolerances: rtol={rtol:.2e}, atol={atol:.2e}")
			print(f"Adjoint method: {use_adjoint}")
			print(f"Time span: {t_span[0].item():.1f}s to {t_span[-1].item():.1f}s ({len(t_span)} points)")
			print(f"Device: {self.device}")
			print()
        
		start_time = time.time()
        
		# Choose between regular and adjoint method
		ode_solver = odeint_adjoint if use_adjoint else odeint
        
		# Solve ODE
		solution = ode_solver(
			self.ode_func,
			initial_state,
			t_span,
			method=method,
			rtol=rtol,
			atol=atol,
			options={'max_num_steps': 100000}
		)
        
		elapsed = time.time() - start_time
        
		if verbose:
			print(f"Integration complete in {elapsed:.2f}s")
			print(f"Solution shape: {solution.shape}")
            
			# Check energy conservation
			r_final = solution[-1, :3]
			v_final = solution[-1, 3:]
			ke_final = 0.5 * torch.sum(v_final**2)
            
			r_init = solution[0, :3]
			v_init = solution[0, 3:]
			ke_init = 0.5 * torch.sum(v_init**2)
            
			U_final = self.gravity_model.compute_potential(r_final.unsqueeze(0), device=str(self.device))
			U_init = self.gravity_model.compute_potential(r_init.unsqueeze(0), device=str(self.device))
            
			if isinstance(U_final, torch.Tensor):
				U_final = U_final.item()
				U_init = U_init.item()
            
			E_final = ke_final.item() + U_final
			E_init = ke_init.item() + U_init
            
			energy_drift = (E_final - E_init) / abs(E_init) * 100
			print(f"Energy drift: {energy_drift:.6e}%")
			print()
        
		return solution


def compare_methods(gravity_model, initial_state, t_hours=3.0):
	"""Compare different torchdiffeq integration methods."""
    
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"Device: {device.upper()}\n")
    
	# Time span
	n_points = 50
	t_span = torch.linspace(0, t_hours * 3600, n_points, device=device)
    
	methods_to_test = [
		('dopri5', 1e-7, 1e-9, 'Adaptive RK45'),
		('dopri8', 1e-9, 1e-11, 'Adaptive RK78'),
		('rk4', 1e-3, 1e-5, 'Fixed RK4'),
	]
    
	results = {}
    
	print("="*80)
	print("COMPARING TORCHDIFFEQ METHODS")
	print("="*80)
	print()
    
	for method, rtol, atol, description in methods_to_test:
		print(f"\n{'─'*80}")
		print(f"Testing: {description} ({method})")
		print(f"{'─'*80}")
        
		sim = TorchDiffEqOrbitSimulator(gravity_model, device=device)
        
		start = time.time()
		solution = sim.propagate(
			initial_state,
			t_span,
			method=method,
			rtol=rtol,
			atol=atol,
			use_adjoint=False,  # For comparison, don't use adjoint
			verbose=True
		)
		elapsed = time.time() - start
        
		results[method] = {
			'solution': solution,
			'time': elapsed,
			'description': description
		}
    
	return results, t_span


def test_gradient_computation(gravity_model, initial_state, target_position):
	"""
	Demonstrate gradient-based optimization using adjoint method.
	Optimize initial velocity to reach target position.
	"""
    
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print("="*80)
	print("GRADIENT-BASED OPTIMIZATION WITH ADJOINT METHOD")
	print("="*80)
	print()
    
	# Make initial velocity optimizable
	r0 = initial_state[:3].clone().detach().to(device)
	v0 = initial_state[3:].clone().detach().to(device).requires_grad_(True)
    
	print(f"Target position: {target_position.numpy()}")
	print(f"Initial velocity: {v0.data.cpu().numpy()}")
	print()
    
	# Time span
	t_span = torch.tensor([0.0, 3600.0], device=device)  # 1 hour
    
	sim = TorchDiffEqOrbitSimulator(gravity_model, device=device)
    
	# Optimizer
	optimizer = torch.optim.Adam([v0], lr=1e-4)
    
	n_iterations = 20
    
	print("Optimizing...")
	print(f"{'Iter':<6} {'Loss':<15} {'Final X':<12} {'Final Y':<12} {'Final Z':<12}")
	print("-"*80)
    
	for i in range(n_iterations):
		optimizer.zero_grad()
        
		# Propagate
		state = torch.cat([r0, v0])
		solution = sim.propagate(
			state,
			t_span,
			method='dopri5',
			rtol=1e-6,
			atol=1e-8,
			use_adjoint=True,  # Use adjoint for efficient gradients!
			verbose=False
		)
        
		# Loss: distance from target
		final_pos = solution[-1, :3]
		loss = torch.sum((final_pos - target_position.to(device))**2)
        
		# Backward pass (uses adjoint method!)
		loss.backward()
        
		# Update
		optimizer.step()
        
		if (i + 1) % 5 == 0 or i == 0:
			print(f"{i+1:<6} {loss.item():<15.6e} {final_pos[0].item():<12.4f} {final_pos[1].item():<12.4f} {final_pos[2].item():<12.4f}")
    
	print()
	print(f"Optimized velocity: {v0.data.cpu().numpy()}")
	print(f"Final position: {final_pos.data.cpu().numpy()}")
	print(f"Target position: {target_position.numpy()}")
	print(f"Error: {torch.sqrt(loss).item():.6f} km")
	print()
	print("✓ Gradients computed using O(1) memory adjoint method!")
	print()

