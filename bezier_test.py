import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.spatial import ConvexHull

def bezier_curve(t, points):
    """Compute a Bezier curve given control points."""
    n = len(points) - 1
    return sum(comb(n, i) * (1 - t) ** (n - i) * t ** i * p for i, p in enumerate(points))

# Define convex anchor points (e.g., vertices of a convex polytope)
num_anchors = 6
anchor_points = np.random.uniform(-3, 3, (num_anchors, 3))  # 6 random points forming a convex hull

# Generate 10 trajectories within the convex hull
num_trajectories = 10
num_control_points = 4  # Bezier curve control points
t_values = np.linspace(0, 1, 100)  # Bezier parameter

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for _ in range(num_trajectories):
    # Generate control points as convex combinations of the anchor points
    weights = np.random.dirichlet(np.ones(num_anchors), size=num_control_points)  # Sum of each row is 1
    control_points = np.dot(weights, anchor_points)  # Weighted sum of anchor points
    
    # Compute Bezier curve
    x = bezier_curve(t_values, control_points[:, 0])
    y = bezier_curve(t_values, control_points[:, 1])
    z = bezier_curve(t_values, control_points[:, 2])

    # Plot trajectory
    ax.plot(x, y, z, label=f"Trajectory {_+1}")

    # Plot control points
    ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], c='black', marker='o')

# Plot the convex anchor points
ax.scatter(anchor_points[:, 0], anchor_points[:, 1], anchor_points[:, 2], c='red', marker='^', s=80, label="Convex Hull Vertices")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Bezier Curves Constrained to a Convex Space")
plt.legend()
plt.show()
