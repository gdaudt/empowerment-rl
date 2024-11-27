import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


rectangle_vertices = [
    [-1, 6, -8],  # Vertex 1
    [-1, 6, 8],  # Vertex 2
    [12, 6, 8],  # Vertex 3
    [12, 6, -8]   # Vertex 4
]

# Create the polygon
verts = [rectangle_vertices]

# Create the polygon collection
collection = Poly3DCollection(verts, facecolors='cyan', edgecolors='r', linewidths=1, alpha=0.5)




# Define the time array
t_start = 0
t_end = 2
num_points = 1000
t = np.linspace(t_start, t_end, num_points)

# Mass of the point particle
m = 15.0  # kg

F_max = 20*9.81*2

# Numer of chebychev basisfunctions
N = 10

# Initial conditions
x0, y0, z0 = 0.0, 0.0, 0.0  # Initial position
v0x, v0y, v0z = 3, 3, 0  # Initial velocity

# Define force functions as functions of time
def rnd_simplex():
  ''' Return uniformly random vector in the n-simplex '''
  k = np.random.exponential(scale=1.0, size=N)
  return k / sum(k)

#np.insert(rnd_simplex(),0,0)

trajs = 200
x_traj = []
y_traj = []
z_traj = []
final_x_in = []
final_y_in = []
final_z_in = []
final_x_out = []
final_y_out = []
final_z_out = []
no_collision = 0

fig = plt.figure(figsize=(30, 25))
ax = fig.add_subplot(111, projection='3d')

final_points = None


x = F_max/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=N)-1)/(N/3), domain=[0, t_end], window=[-1,1])(t)
# plt.plot(t,x)
# plt.show()

for _ in range(trajs):

  F_x = F_max/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=N)-1)/(N/3), domain=[0, t_end], window=[-1,1])
  F_y = F_max/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=N)-1)/(N/3), domain=[0, t_end], window=[-1,1])
  F_z = F_max/4*np.polynomial.chebyshev.Chebyshev(coef=(2*np.random.uniform(size=N)-1)/(N/3), domain=[0, t_end], window=[-1,1])

  # Compute accelerations
  a_x = F_x(t) / m
  a_y = F_y(t) / m
  a_z = F_z(t) / m

  # Numerically integrate acceleration to get velocity
  v_x = v0x + cumulative_trapezoid(a_x, t, initial=0)
  v_y = v0y + cumulative_trapezoid(a_y, t, initial=0)
  v_z = v0z + cumulative_trapezoid(a_z, t, initial=0)

  # Numerically integrate velocity to get position
  x = x0 + cumulative_trapezoid(v_x, t, initial=0)
  y = y0 + cumulative_trapezoid(v_y, t, initial=0)
  z = z0 + cumulative_trapezoid(v_z, t, initial=0)

  x_traj.append(x)
  y_traj.append(y)
  z_traj.append(z)

  if y[-1] > 6:
    final_x_out.append(x[-1])
    final_y_out.append(y[-1])
    final_z_out.append(z[-1])
  else:
    final_x_in.append(x[-1])
    final_y_in.append(y[-1])
    final_z_in.append(z[-1])
    if final_points is None:
      final_points = np.array([x[-1], y[-1], z[-1]])
    else:
      final_points = np.vstack((final_points, np.array([x[-1], y[-1], z[-1]])))
    no_collision += 1

  ax.plot(x, y, z, color='blue', linewidth=0.2)

print(np.shape(final_points))
hull = ConvexHull(final_points)

Empowerment = np.log(hull.volume)
print('Empowerment:', Empowerment)

#print(no_collision/trajs)

# Plot the trajectory in 3D space



ax.scatter(final_x_in, final_y_in, final_z_in, marker = '^', s=20, color='green')
ax.scatter(final_x_out, final_y_out, final_z_out, marker = '^', s=20, color='red')

ax.scatter(0, 0, 0, marker = '^', s=30, color='yellow')

ax.add_collection3d(collection)

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory of a Point Mass under Applied Forces')
ax.legend()
plt.show()