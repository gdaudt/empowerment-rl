import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull


class IndexedPoint:
    def __init__(self, coords, index):
        self.coords = np.array(coords)
        self.index = index

    def __getitem__(self, key):
        return self.coords[key]
    
    def __len__(self):
        return len(self.coords)
    
    def __repr__(self):
        return f"IndexedPoint({self.coords}, {self.index})"
    
    def __lt__(self, other):
        if isinstance(other, IndexedPoint):
            return tuple(self.coords) < tuple(other.coords)
        return NotImplemented
    
    def get_index(self):
        return self.index
    
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)
        if rootA != rootB:
            self.parent[rootB] = rootA
            
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
x0, y0, z0 = 0.0, 0.0, 0.0  # Initial position
v0x, v0y, v0z = 0.0, 0.0, 0.0  # Initial velocity

#set of obstacles with center position and size. two obstacles with size 1, one centered at (0, 1, 0), and the other one t (-2, 1, 0)
obs_centers = [(0, 3, 0), (-2, 1, 0)]
obstacle_size = (1, 3, 1)


# Number of trajectories to generate
num_trajectories = 300

# Container for storing trajectory directions (in radians)
trajectory_angles = []

# Containers for final points
all_final_points = []    # All final trajectory points
colliding_points = []    # Final points colliding with the obstacle

def check_collision(point):
    for center in obs_centers:
        if dim == 2:
            if (center[0] - obstacle_size[0] / 2 <= point[0] <= center[0] + obstacle_size[0] / 2 and
                center[1] - obstacle_size[1] / 2 <= point[1] <= center[1] + obstacle_size[1] / 2):
                return True
        else:
            if (center[0] - obstacle_size[0] / 2 <= point[0] <= center[0] + obstacle_size[0] / 2 and
                center[1] - obstacle_size[1] / 2 <= point[1] <= center[1] + obstacle_size[1] / 2 and
                center[2] - obstacle_size[2] / 2 <= point[2] <= center[2] + obstacle_size[2] / 2):
                return True
    return False


# --- Plot setup ---
if dim == 2:
    fig, ax = plt.subplots(figsize=(10, 7))
    #plot the obstacles in 2D
    for center in obs_centers:
        ax.add_patch(plt.Rectangle((center[0] - obstacle_size[0] / 2, center[1] - obstacle_size[1] / 2),
                                   obstacle_size[0], obstacle_size[1], fill=True, color='gray'))
else:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

# --- Trajectory generation ---

uf = UnionFind()

for i in range(num_trajectories):
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

    # Determine the horizontal angle (in radians) relative to the positive x-axis
    angle = np.arctan2(dy, dx)
    if angle < 0:
        angle = 2 * np.pi + angle
    trajectory_angles.append(angle)

    # Determine final point and append it to the overall list
    if dim == 2:
        final_point = (x[-1], y[-1])
    else:
        final_point = (x[-1], y[-1], z[-1])
    
    #initialize the uf element for this point
    uf.parent[i] = i
    all_final_points.append(final_point)
    
    
    # --- Plotting the trajectory ---
    plot_colliding_only = True
    
    if plot_colliding_only:
        if check_collision(final_point):
            if dim == 2:
                ax.plot(x, y, linewidth=1)
                #plot the final point with a dot colored red if it collides with an obstacle, and blue otherwise
            
                
            else:
                
                ax.plot(x, y, z, linewidth=1)   
                #plot the final point with a dot colored red if it collides with an obstacle, and blue otherwise
            

# create a kd-tree containing all final points
kd_tree = cKDTree(all_final_points)
visited = set()
components = []

for i, point in enumerate(all_final_points):
    if not check_collision(point) or i in visited:
        continue
    
    uf.find(i)
    component = []
    stack = [i]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        # get the nearest 4 neighbours of the current point that have not been visited
        neighbours = kd_tree.query(point, k=4)[1]        
        for neighbour in neighbours:
            #component.append(kd_tree.data[neighbour])
            component.append(neighbour)
            uf.union(current, neighbour)
            visited.add(neighbour)
            
    components.append(component)
#print("components: ", components)   

merged_components = {}
for index in visited:
    #print("index: ", index)
    root = uf.find(index)
    #print("root: ", root)
    if root not in merged_components:
        merged_components[root] = set()
    merged_components[root].add(index)
    # print("merged_components[root]: ", merged_components[root])
    # print("merged components: ", merged_components)
    
#print("merged components: ", merged_components)


# plot the final points of the trajectories as dots colored blue
# for point in all_final_points:
#     if dim == 2:
#         ax.plot(point[0], point[1], 'bo')
#     else:
#         ax.plot(point[0], point[1], point[2], 'bo')

#color each set of merged_components a different random color
colors = np.random.rand(len(merged_components), 3)
for i, component in enumerate(merged_components.values()):
    #access the value through the kd_tree using kd_tree.data[value]
    component = [kd_tree.data[value] for value in component]
    component = np.array(component)
    if dim == 2:
        ax.plot(component[:, 0], component[:, 1], 'o', color=colors[i])
    else:
        ax.plot(component[:, 0], component[:, 1], component[:, 2], 'o', color=colors[i])
    

    #if they are in the components list, color them a different color for each component

# --- Labeling and showing plot ---
trajectory_angles = np.array(trajectory_angles)
# print("Trajectory directions (radians):")
# print(trajectory_angles)
# print("All final points:")
# print(all_final_points)
# print("Colliding points:")
# print(colliding_points)

#calculate the convex hull of all final points
hull = ConvexHull(all_final_points)
print("hull vertices: ", hull.vertices)
#plot the convex hull area in 2d, in green, with alpha set to 0.5
hull_points = []
for vertices in hull.vertices:
    hull_points.append(all_final_points[vertices])
hull_points = np.concatenate([hull_points, hull_points[:1]], axis=0)

# Plot the convex hull boundary
plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label='Convex Hull Boundary')

# Fill the area inside the convex hull
plt.fill(hull_points[:, 0], hull_points[:, 1], 'lightblue', alpha=0.3,
         label=f'Hull Area: {hull.volume:.2f}')

# now calculate the convex hull of each component in the components list ----- OLD VERSION NO MERGING
# for component in components:
#     component_hull = ConvexHull(component)
#     component_hull_points = []
#     for vertices in component_hull.vertices:
#         component_hull_points.append(component[vertices])
#     component_hull_points = np.concatenate([component_hull_points, component_hull_points[:1]], axis=0)
#     plt.plot(component_hull_points[:, 0], component_hull_points[:, 1], 'r--', lw=2, label='Convex Hull Boundary')
#     plt.fill(component_hull_points[:, 0], component_hull_points[:, 1], 'red', alpha=0.4,
#          label=f'Hull Area: {component_hull.volume:.2f}')

# calculate the convex hull of each component in the merged_components list
for component in merged_components.values():
    component = [kd_tree.data[value] for value in component]
    component_hull = ConvexHull(component)
    component_hull_points = []
    for vertices in component_hull.vertices:
        component_hull_points.append(component[vertices])
    component_hull_points = np.concatenate([component_hull_points, component_hull_points[:1]], axis=0)
    plt.plot(component_hull_points[:, 0], component_hull_points[:, 1], 'r--', lw=2, label='Convex Hull Boundary')
    plt.fill(component_hull_points[:, 0], component_hull_points[:, 1], 'red', alpha=0.4,
         label=f'Hull Area: {component_hull.volume:.2f}')

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


