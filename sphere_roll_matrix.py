import numpy as np


def sphere_roll(array: np.ndarray, rotation: np.ndarray, get_azel=False, axises=[0, 1]):
    """
    Arbitrary rotation matrix spanned by a sphere (sphere coordinate system)

    :param array: np.ndarray dimention >= 2
    :param rotation: shape: rotation matrix - (3, 3); euler_angles - (3,), quaternion - (4,), self rotate - (4, 1)
    :param get_azel: to return new_array, Azimuth, Elevation
    :param axises: list of two elements. First - elevation exis, second - azimuth axis
    :return:
    """
    # -------------create rotation matrix----------------
    if rotation.shape == (3, 3):  # rotation matrix
        rot_mat = rotation
    elif rotation.shape == (3,):  # euler angles (radians)
        [phi_x, phi_y, phi_z] = rotation
        M_x = np.array([[1, 0, 0],
                        [0, np.cos(phi_x), -np.sin(phi_x)],
                        [0, np.sin(phi_x), np.cos(phi_x)]])
        M_y = np.array([[np.cos(phi_y), 0, np.sin(phi_y)],
                        [0, 1, 0],
                        [-np.sin(phi_y), 0, np.cos(phi_y)]])
        M_z = np.array([[np.cos(phi_z), -np.sin(phi_z), 0],
                        [np.sin(phi_z), np.cos(phi_z), 0],
                        [0, 0, 1]])
        rot_mat = M_x @ M_y @ M_z
    elif rotation.shape == (4,) or rotation.shape == (4, 1):
        if rotation.shape == (4, 1):  # self rotation
            [w, x, y, z] = rotation.flatten()  # w[rads] - angle, [x,y,z] - axis for rotation
            qw = np.cos(w / 2)
            [qx, qy, qz] = np.array([x, y, z]) * np.sin(w / 2)
            rotation = np.array([qw, qx, qy, qz])

        [w, x, y, z] = rotation  # quaternion
        rot_mat = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    else:
        raise ValueError('Incorrect rotation')
    # ---------------------------------------------------

    n_el, n_az = array.shape[axises[0]], array.shape[axises[1]]

    AZ, EL = np.meshgrid(np.linspace(-np.pi, np.pi, n_az, endpoint=False),
                         np.linspace(-np.pi / 2, np.pi / 2, n_el, endpoint=False))

    # -----------------------Sphere2Cart----------------------------------
    cos_theta = np.cos(EL)
    x = cos_theta * np.cos(AZ)
    y = cos_theta * np.sin(AZ)
    z = np.sin(EL)
    # -------------------Rotation process---------------------------------
    [x1, y1, z1] = np.apply_along_axis(lambda vec3: vec3 @ rot_mat, 0, np.array([x, y, z]))

    # -----------------------Cart2Sphere----------------------------------
    hxy = np.hypot(x1, y1)
    el = np.arctan2(z1, hxy)
    az = np.arctan2(y1, x1)
    # --------------------------Reindexing--------------------------------

    ind_az = np.round((az + np.pi) / (2 * np.pi) * (n_az - 1)).astype(int)
    ind_el = np.round((el + np.pi / 2) / (np.pi) * (n_el - 1)).astype(int)

    slices = [slice(None)] * len(array.shape)
    slices[axises[0]] = ind_el.flatten()
    slices[axises[1]] = ind_az.flatten()

    new_array = array[np.s_[tuple(slices)]].reshape(array.shape)
    if get_azel:
        return [new_array, AZ, EL]
    return new_array
