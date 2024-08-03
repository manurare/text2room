import numpy as np
import torch
from model.mesh_fusion.util import get_extrinsics


#######################
# PRIVATE HELPERS #
#######################

def _rot_x(x):
    '''
    positive: look up, negative: look down

    :param x: rotation in degrees
    '''
    return np.array([x * np.pi / 180, 0, 0], dtype=np.float32)


def _rot_y(x):
    '''
    positive: look right, negative: look left

    :param x: rotation in degrees
    '''
    return np.array([0, x * np.pi / 180, 0], dtype=np.float32)


def _rot_z(x):
    '''
    positive: tilt left, negative: tilt right

    :param x: rotation in degrees
    '''
    return np.array([0, 0, x * np.pi / 180], dtype=np.float32)


def _trans_x(x):
    '''
    positive: right, negative: left

    :param x: translation amount
    '''
    return np.array([x, 0, 0], dtype=np.float32)


def _trans_y(x):
    '''
    positive: down, negative: up

    :param x: translation amount
    '''
    return np.array([0, x, 0], dtype=np.float32)


def _trans_z(x):
    '''
    positive: back, negative: front

    :param x: translation amount
    '''
    return np.array([0, 0, x], dtype=np.float32)


def _config_fn(fn, **kwargs):
    return lambda i, steps: fn(i, steps, **kwargs)


def _circle(i, steps=60, txmax=0, txmin=0, tymax=0, tymin=0, tzmax=0, tzmin=0, rxmax=0, rxmin=0, rymax=0, rymin=0, rzmax=0, rzmin=0):
    tx_delta = (txmax - txmin) / (steps // 2)
    ty_delta = (tymax - tymin) / (steps // 2)
    tz_delta = (tzmax - tzmin) / (steps // 2)

    rx_delta = (rxmax - rxmin) / (steps // 2)
    ry_delta = (rymax - rymin) / (steps // 2)
    rz_delta = (rzmax - rzmin) / (steps // 2)

    f = i % (steps // 2)

    tx = txmin + f * tx_delta
    ty = tymin + f * ty_delta
    tz = tzmin + f * tz_delta

    rx = rxmin + f * rx_delta
    ry = rymin + f * ry_delta
    rz = rzmin + f * rz_delta

    if i < steps // 2:
        T = _trans_x(-tx)
        T += _trans_z(tz)
        T += _trans_y(ty)
        R = _rot_y(ry)
        R += _rot_x(rx)
        R += _rot_z(rz)
    else:
        T = _trans_x(tx)
        T += _trans_z(tz)
        T += _trans_y(-ty)
        R = _rot_y(-ry)
        R += _rot_x(-rx)
        R += _rot_z(-rz)

    return get_extrinsics(R, T)


def _rot_left(i, steps=60, ty=0, rx=0):
    angle = i * 360 // steps

    T = _trans_x(0)
    T += _trans_y(ty)
    R = _rot_y(-angle)
    R += _rot_x(rx)

    return get_extrinsics(R, T)


def _lemniscate(i, n_steps=255, **args):
    # Parameters
    a = 0.5  # Scale of the lemniscate

    # Create theta values
    theta1 = np.linspace(0.5*np.pi, 0.75*np.pi, 25)
    theta2 = np.linspace(0.75*np.pi, 1.25*np.pi, 80)
    theta3 = np.linspace(1.25*np.pi, 1.75*np.pi, 45)
    theta4 = np.linspace(1.75*np.pi, 2.25*np.pi, 80)
    theta5 = np.linspace(2.25*np.pi, 2.5*np.pi, 25)
    theta = np.concatenate((theta1, theta2, theta3, theta4, theta5))

    # Convert to Cartesian coordinates
    x = a * np.cos(theta) / (np.sin(theta)**2 + 1)
    z = a * np.cos(theta) * np.sin(theta) / (np.sin(theta)**2 + 1)
    y = a * 0.2 * np.cos(4*theta)
    Cs = np.stack((x, y, z)).T

    dx_dtheta = -a*(np.sin(theta) * (np.sin(theta) ** 2 + 2*np.cos(theta)**2 + 1)) / (np.sin(theta)**2 + 1)**2
    dz_dtheta = -a*(np.sin(theta)**4 + np.sin(theta)**2 + (np.sin(theta)**2-1)*np.cos(theta)**2) / (np.sin(theta)**2 + 1)**2

    lookat = np.stack((dx_dtheta, np.zeros_like(x), dz_dtheta)).T
    lookat = lookat / np.linalg.norm(lookat, axis=1, keepdims=True)

    UP = np.array([0, 1, 0])[None, :]
    right = np.cross(UP, lookat)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(lookat, right)
    up /= np.linalg.norm(up, axis=1, keepdims=True)

    Rs = np.concatenate((right, up, lookat), axis=1).reshape(-1, 3, 3)

    R = torch.from_numpy(Rs[i]) 
    T = -(R @ torch.from_numpy(Cs[i]))

    RT = torch.cat([R, T[:, None]], dim=-1).to("cpu")  # RT is [4,4]
    RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
    return RT


def _back_and_forth(i, steps=20, txmax=0, txmin=0, tymax=0, tymin=0, tzmax=0, tzmin=0, rxmax=0, rxmin=0, rymax=0, rymin=0, rzmax=0, rzmin=0):
    tx_delta = (txmax - txmin) / (steps // 2)
    ty_delta = (tymax - tymin) / (steps // 2)
    tz_delta = (tzmax - tzmin) / (steps // 2)

    rx_delta = (rxmax - rxmin) / (steps // 2)
    ry_delta = (rymax - rymin) / (steps // 2)
    rz_delta = (rzmax - rzmin) / (steps // 2)

    f = i % (steps // 2)

    tx = txmin + f * tx_delta
    ty = tymin + f * ty_delta
    tz = tzmin + f * tz_delta

    rx = rxmin + f * rx_delta
    ry = rymin + f * ry_delta
    rz = rzmin + f * rz_delta

    if i < steps // 2:
        T = _trans_x(-tx)
        T += _trans_z(tz)
        T += _trans_y(ty)
        R = _rot_y(-ry)
        R += _rot_x(rx)
        R += _rot_z(rz)
    else:
        T = _trans_x(tx)
        T += _trans_z(tz)
        T += _trans_y(-ty)
        R = _rot_y(ry)
        R += _rot_x(-rx)
        R += _rot_z(-rz)

    return get_extrinsics(R, T)


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    c2w[0:3, 3] = -1 * c2w[0:3, 3]
    return c2w


# def sample_points_on_unit_sphere(n_sample, radius=4.0, height=0.0):
#     c2w = torch.stack([pose_spherical(angle, 0, radius) for angle in np.linspace(-180,180,n_sample)], dim=0)
#     c2w[:, 1, 3] = height
#     return c2w

# def spherical_trajector(steps, radius, height):
#     c2w = sample_points_on_unit_sphere(n_sample, radius, height)
#     return torch.inverse(c2w)


def _sphere_rot_xz(i, steps, radius=4.0, height=0.0, phi=20.0):
    rot_angle = i * 360 / steps
    rot_angle = rot_angle
    phi = phi if height > 0 else -phi if height < 0 else 0
    c2w = pose_spherical(rot_angle, phi, radius)
    c2w[1, 3] = height
    return torch.inverse(c2w)

def _double_sphere_rot_xz(i, steps, radius=4.0, height=0.0, phi=20.0):
    rot_angle = i * 360 / steps
    rot_angle = rot_angle
    phi = phi if height > 0 else -phi if height < 0 else 0
    c2w = pose_spherical(rot_angle, phi, radius)
    if i % 2 == 0:
        c2w[0:3, 3] = 0
    else:
        c2w[1, 3] = height
    return torch.inverse(c2w)

#######################
# PUBLIC TRAJECTORIES #
#######################

def lemniscate(**args):
    return _config_fn(_lemniscate, **args)


def forward(height=0, rot=0, txmax=2):
    return _config_fn(_circle, txmax=txmax, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def forward_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def left_right(height=0, rot=0):
    return _config_fn(_circle, tzmax=2, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward(height=0, rot=0):
    return _config_fn(_circle, txmax=-2, rymax=90, rymin=45, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward2(height=0, rot=0, txmax=1):
    return _config_fn(_circle, txmax=txmax, rymax=225, rymin=180, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward2_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=225, rymin=180, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward3(height=0, rot=0, txmax=1):
    return _config_fn(_circle, txmax=txmax, rymax=270, rymin=225, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def backward3_small(height=0, rot=0):
    return _config_fn(_circle, txmax=0.5, rymax=270, rymin=225, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def rot_left_up_down(height=0, rot=0):
    return _config_fn(_rot_left, ty=height, rx=rot)


def back_and_forth_forward(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=-2, rymax=60, rymin=15, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=-2, rymax=-120, rymin=-165, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_forward_reverse(height=0, rot=0, tzmax=2):
    return _config_fn(_back_and_forth, txmax=0, tzmax=tzmax, rymax=-15, rymin=-60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_forward_reverse_small(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=0.5, rymax=-15, rymin=-60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse(height=0, rot=0, tzmax=2):
    return _config_fn(_back_and_forth, txmax=0, tzmax=tzmax, rymax=165, rymin=120, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse_small(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=0.5, rymax=165, rymin=120, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def back_and_forth_backward_reverse2(height=0, rot=0):
    return _config_fn(_back_and_forth, txmax=0, tzmax=3, rymax=165, rymin=60, tymin=height, tymax=height, rxmin=rot, rxmax=rot)


def sphere_rot(radius=4.0, height=0.0, phi=20.0):
    return _config_fn(_sphere_rot_xz, radius=radius, height=height, phi=phi)


def double_rot(radius=4.0, height=0.0, phi=2.0):
    return _config_fn(_double_sphere_rot_xz, radius=radius, height=height, phi=phi)
