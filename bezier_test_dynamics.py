import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Constants
m = 0.027  # kg
g = 9.81  # m/s²
max_thrust = 2.33  # m/s² (additional thrust)
F_max = m * (g + max_thrust)  # Maximum force

# Time range
t_end = 2  # Total duration

# Bézier basis function
def bernstein(n, i, t):
    """ Compute the Bernstein polynomial B_i^n(t). """
    return comb(n, i) * (t**i) * ((1 - t) ** (n - i))

# Bézier curve computation
def bezier(control_points, t):
    """ Compute the Bézier curve at t given control points. """
    n = len(control_points) - 1
    return sum(p * bernstein(n, i, t) for i, p in enumerate(control_points))

# Number of trajectories
num_trajectories = 10
n = 3  # Degree of the Bézier curve

# Initial conditions (set as needed)
x0, y0, z0 = 0.0, 0.0, 0.0  # Initial position
v0x, v0y, v0z = 0.0, 0.0, 0.0  # Initial velocity

# Set up 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate and plot multiple trajectories
for _ in range(num_trajectories):
    # Randomized Bézier control points for force (bounded by F_max)
    control_points_x = np.random.uniform(-F_max, F_max, n+1)
    control_points_y = np.random.uniform(-F_max, F_max, n+1)
    control_points_z = np.random.uniform(-F_max, F_max, n+1)

    # Compute velocity control points (integral of force/mass)
    Q_x = np.cumsum(control_points_x / m) / (n + 1)
    Q_y = np.cumsum(control_points_y / m) / (n + 1)
    Q_z = np.cumsum(control_points_z / m) / (n + 1)

    # Compute position control points (integral of velocity)
    R_x = np.cumsum(Q_x) / (n + 2)
    R_y = np.cumsum(Q_y) / (n + 2)
    R_z = np.cumsum(Q_z) / (n + 2)

    # Generate time samples
    t_values = np.linspace(0, t_end, 100)

    # Compute trajectory points
    x_traj = x0 + v0x * t_values + np.array([bezier(R_x, t) for t in t_values])
    y_traj = y0 + v0y * t_values + np.array([bezier(R_y, t) for t in t_values])
    z_traj = z0 + v0z * t_values + np.array([bezier(R_z, t) for t in t_values])

    # Plot the trajectory
    ax.plot(x_traj, y_traj, z_traj, linewidth=1)

# Labels and formatting
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("3D Bézier-Based Trajectories")
plt.show()
