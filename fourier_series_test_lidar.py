import numpy as np
import matplotlib.pyplot as plt

# --- User parameters ---
dim = 2  # Set to 2 for 2D trajectories or 3 for 3D trajectories

# Constants
m = 0.027  # kg
g = 9.81  # m/s² (gravity)
max_thrust = 8.33  # m/s² (max additional thrust)
F_max = m * (g + max_thrust)  # Maximum force

# Time parameters
t_end = 1  # seconds
num_samples = 100  # Number of time steps
t = np.linspace(0, t_end, num_samples)

# Fourier series parameters
N = 5  # Number of Fourier terms
omega = 2 * np.pi / t_end  # Base frequency (one full oscillation per time window)

# Initial conditions (customize as needed)
# For 2D, only x0, y0 and v0x, v0y are used.
x0, y0, z0 = 0.0, 0.0, 0.0  # Initial position
v0x, v0y, v0z = 3.0, 0.0, 0.0  # Initial velocity

# Number of trajectories to generate
num_trajectories = 10

# Container for storing trajectory directions (in radians)
trajectory_angles = []
trajectory_distances = []

#LiDAR parameters
num_rays = 360
HORIZONTAL_ANGLE = 2 * np.pi  # full 360° sweep
# Create the beam angles (in radians)
beam_angles = np.linspace(0, HORIZONTAL_ANGLE, num_rays, endpoint=False)
# Generate random LiDAR distance readings (in the same distance units as trajectories)
lidar_readings = np.random.uniform(0.5, 3, num_rays)
print("Lidar readings:", lidar_readings)

# --- Plot setup ---
if dim == 2:
    fig, ax = plt.subplots(figsize=(10, 7))
else:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

# --- Trajectory generation ---
for _ in range(num_trajectories):
    # Random Fourier coefficients within force limits for x and y (always used)
    A_x = np.random.uniform(-F_max, F_max, N)
    B_x = np.random.uniform(-F_max, F_max, N)
    A_y = np.random.uniform(-F_max, F_max, N)
    B_y = np.random.uniform(-F_max, F_max, N)

    # Compute acceleration as Fourier series for x and y (divide by mass to get acceleration)
    a_x = np.sum([A_x[n] * np.cos((n+1) * omega * t) + B_x[n] * np.sin((n+1) * omega * t)
                  for n in range(N)], axis=0) / m
    a_y = np.sum([A_y[n] * np.cos((n+1) * omega * t) + B_y[n] * np.sin((n+1) * omega * t)
                  for n in range(N)], axis=0) / m

    # For 3D, also compute for z
    if dim == 3:
        A_z = np.random.uniform(-F_max, F_max, N)
        B_z = np.random.uniform(-F_max, F_max, N)
        a_z = np.sum([A_z[n] * np.cos((n+1) * omega * t) + B_z[n] * np.sin((n+1) * omega * t)
                      for n in range(N)], axis=0) / m

    # Integration time-step
    dt = t_end / num_samples

    # Integrate acceleration to get velocity
    v_x = v0x + np.cumsum(a_x) * dt
    v_y = v0y + np.cumsum(a_y) * dt
    if dim == 3:
        v_z = v0z + np.cumsum(a_z) * dt

    # Integrate velocity to get position
    x = x0 + np.cumsum(v_x) * dt
    y = y0 + np.cumsum(v_y) * dt
    if dim == 3:
        z = z0 + np.cumsum(v_z) * dt

    # Compute overall displacement in the horizontal plane
    dx = x[-1] - x0
    dy = y[-1] - y0
    if dim == 3:
        dz = z[-1] - z0
    # The angle (in radians) relative to the positive x-axis:
    if dim == 2:
        angle = np.arctan2(dy, dx)
    else:
        angle = np.arctan2(dy, dx)
        # For 3D, we can also compute the vertical angle
        angle_z = np.arctan2(dz, dx)
        print(f"Horizontal angle: {angle:.2f} radians, Vertical angle: {angle_z:.2f} radians")
    #convert the angle to positive radians from 0 to 2pi
    if angle < 0:
        angle = 2 * np.pi + angle
    trajectory_angles.append(angle)
    traj_distance = np.hypot(dx, dy)
    trajectory_distances.append(traj_distance)
    # Plot the trajectory based on the dimension
    if dim == 2:
        ax.plot(x, y, linewidth=1)
        # Optionally, mark the final point with an arrow indicating the direction
        ax.arrow(x[-1], y[-1], 0.1 * np.cos(angle), 0.1 * np.sin(angle), 
                 head_width=0.05, head_length=0.1, fc='r', ec='r')
        # Also plot the arrow from the starting position
        ax.arrow(x0, y0, 0.1 * np.cos(angle), 0.1 * np.sin(angle), 
                 head_width=0.05, head_length=0.1, fc='b', ec='b')
    else:
        ax.plot(x, y, z, linewidth=1)
        # Optionally, mark the final point with an arrow indicating the direction
        # For 3D we have to combine the horizontal and vertical angles
        ax.quiver(x[-1], y[-1], z[-1], np.cos(angle) * np.cos(angle_z), np.sin(angle) * np.cos(angle_z), np.sin(angle_z),
                  color='red', length=0.2, normalize=True)
        # Also plot the arrow from the starting position
        ax.quiver(x0, y0, z0, np.cos(angle) * np.cos(angle_z), np.sin(angle) * np.cos(angle_z), np.sin(angle_z),
                  color='blue', length=0.2, normalize=True)

for traj_index, (traj_angle, traj_distance) in enumerate(zip(trajectory_angles, trajectory_distances)):
    
    beam_index = np.argmin(np.abs(beam_angles - traj_angle))
    print("For trajectory", traj_index, "the closest beam is", beam_index)
    beam_distance = lidar_readings[beam_index]
    if traj_distance > beam_distance:
        print(f"Trajectory {traj_index} is colliding with beam {beam_index} of angle {beam_angles[beam_index]}: "
              f"angle = {traj_angle:.2f} rad, trajectory distance = {traj_distance:.2f}, "
              f"beam reading = {beam_distance:.2f}")
    else:
        print(f"Trajectory {traj_index} is clear in beam {beam_index} of angle {beam_angles[beam_index]}: "
              f"angle = {traj_angle:.2f} rad, trajectory distance = {traj_distance:.2f}, "
              f"beam reading = {beam_distance:.2f}")

# Plot the lidar beams as straight lines according to the readings
if dim == 2:
    for beam_angle, beam_distance in zip(beam_angles, lidar_readings):
        x_beam = [0, beam_distance * np.cos(beam_angle)]
        y_beam = [0, beam_distance * np.sin(beam_angle)]
        ax.plot(x_beam, y_beam, linestyle='--', color='gray', linewidth=0.5)  

# --- Labeling and showing plot ---
# --- Output trajectory angles ---
trajectory_angles = np.array(trajectory_angles)
print("Trajectory directions (radians):")
print(trajectory_angles)

if dim == 2:
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("2D Fourier Series-Based Trajectories")
else:
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("3D Fourier Series-Based Trajectories")

plt.show()

