import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from sphere_roll_matrix import sphere_roll


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


RP = np.load('antennaRP.npy')

# --------------------------------rot matrix--------------------------------
phi = np.pi / 2
rot_mat = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])  # x axis (rotataton matrix)
# --------------------------------self rot----------------------------------
self_rot = np.array([[np.pi / 2], [0], [1], [0]])  # y axis
# -------------------------------euler rot----------------------------------
euler_rot = np.array([0, 0, np.pi / 2])
# -------------------------------quat rot-----------------------------------
q_phi = np.pi / 4
q = np.array([np.cos(q_phi / 2), *(np.sin(q_phi / 2) * np.array([0.707, 0.707, 0]))])
# --------------------------------------------------------------------------
RP2 = np.array([np.real(RP), np.imag(RP)])
[new_RP1, AZ, EL] = sphere_roll(RP2, rot_mat, True, axises=[1, 2])
new_RP2 = sphere_roll(RP2, self_rot, False, axises=[1, 2])
new_RP3 = sphere_roll(RP2, euler_rot, False, axises=[1, 2])
new_RP4 = sphere_roll(RP2, q, False, axises=[1, 2])

[x1, y1, z1] = sph2cart(AZ, EL, np.sqrt(new_RP1[0]**2 + new_RP1[1]**2))
[x2, y2, z2] = sph2cart(AZ, EL, np.sqrt(new_RP2[0]**2 + new_RP2[1]**2))
[x3, y3, z3] = sph2cart(AZ, EL, np.sqrt(new_RP3[0]**2 + new_RP3[1]**2))
[x4, y4, z4] = sph2cart(AZ, EL, np.sqrt(new_RP4[0]**2 + new_RP4[1]**2))
res = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
# Make data.
# Plot the surface.
surf = ax.plot_surface(*res[3], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
