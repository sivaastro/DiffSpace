import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure standalone_simulator is importable when run directly
ROOT = Path(__file__).resolve().parents[1].parent
sys.path.append(str(ROOT))

# Import models from standalone folder
from standalone_simulator.models.gravity_models_fast import (
    PointMassGravity,
    NagyPolyhedronGravityFast,
    PolyhedronGravityFast,
    SphericalHarmonicsGravityFast
)

# Use .mat files colocated under models folder (standalone bundle)
DATA = Path(__file__).resolve().parents[1] / 'models'

def make_model(name: str):
    a, b, c = 34.4, 11.2, 11.2
    density = 2670e9
    G = 6.674308e-20
    volume = (4/3) * np.pi * (a/2) * (b/2) * (c/2)
    mu = G * volume * density

    if name == 'point':
        return PointMassGravity(mu)
    if name == 'nagy':
        return NagyPolyhedronGravityFast((a, b, c), density)
    if name == 'poly':
        return PolyhedronGravityFast(str(DATA / 'ParaV.mat'), str(DATA / 'ParaF.mat'))
    if name == 'sph':
        return SphericalHarmonicsGravityFast(mu, str(DATA / 'SPH.mat'), 4, (a, b, c))
    raise ValueError('Unknown model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelA', default='nagy', choices=['point','nagy','poly','sph'])
    parser.add_argument('--modelB', default='poly', choices=['point','nagy','poly','sph'])
    parser.add_argument('--xmin', type=float, default=-50)
    parser.add_argument('--xmax', type=float, default=50)
    parser.add_argument('--ymin', type=float, default=-50)
    parser.add_argument('--ymax', type=float, default=50)
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    args = parser.parse_args()

    mA = make_model(args.modelA)
    mB = make_model(args.modelB)

    X = torch.linspace(args.xmin, args.xmax, args.nx)
    Y = torch.linspace(args.ymin, args.ymax, args.ny)
    xv, yv = torch.meshgrid(X, Y, indexing='xy')
    zv = torch.zeros_like(xv)
    pos = torch.stack([xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)], dim=1)

    aA = mA.compute_gravity(pos)
    aB = mB.compute_gravity(pos)

    abs_err = torch.linalg.norm(aA - aB, dim=1).reshape(args.nx, args.ny)

    plt.figure(figsize=(8,6))
    im = plt.imshow(abs_err.numpy().T, origin='lower', extent=[args.xmin,args.xmax,args.ymin,args.ymax], cmap='viridis')
    plt.colorbar(im, label='|a_A - a_B| (km/s^2)')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title(f'Absolute Gravity Error: {args.modelA} vs {args.modelB}')
    out = Path('abs_error_fields.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
