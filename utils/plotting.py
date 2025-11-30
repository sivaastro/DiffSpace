import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_orbit_xy(solution, title='Orbit (XY)', outfile='orbit_xy.png'):
    """Plot 2D XY projection of orbit."""
    r = solution[:, :3].cpu().numpy()
    plt.figure(figsize=(7,6))
    plt.plot(r[:,0], r[:,1], lw=2)
    plt.scatter(0,0,c='orange',s=120,marker='*')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close()

def plot_orbit_3d(solution, title='Orbit (3D)', outfile='orbit_3d.png'):
    """Plot 3D orbit trajectory."""
    r = solution[:, :3].cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[:,0], r[:,1], r[:,2], lw=2)
    ax.scatter(0, 0, 0, c='orange', s=200, marker='*', label='Asteroid')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close()

def plot_orbit_metrics(solution, t_span, title='Orbit Metrics', outfile='orbit_metrics.png'):
    """Plot distance and velocity over time."""
    r = solution[:, :3].cpu().numpy()
    v = solution[:, 3:].cpu().numpy()
    t = t_span.cpu().numpy() / 3600  # Convert to hours
    
    dist = np.linalg.norm(r, axis=1)
    vel = np.linalg.norm(v, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t, dist, lw=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Orbital Radius')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, vel, lw=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.set_title('Velocity Magnitude')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close()
