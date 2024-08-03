import torch
import numpy as np
import open3d as o3d
import trimesh

from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix


def get_pixel_grids(height, width, reverse=False):
    # texture coordinate.
    if reverse:
        # Pytorch3D expects +X left and +Y up!!!
        x_linspace = torch.linspace(width - 1, 0, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(height - 1, 0, height).view(height, 1).expand(height, width)
    else:
        x_linspace = torch.linspace(0, width - 1, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height).view(height, 1).expand(height, width)
    x_coordinates = x_linspace.contiguous().view(-1)
    y_coordinates = y_linspace.contiguous().view(-1)
    ones = torch.ones(height * width)
    indices_grid = torch.stack([x_coordinates,  y_coordinates, ones], dim=0)
    return indices_grid


def img_to_pts(height, width, depth, K=torch.eye(3)):
    pts = get_pixel_grids(height, width).to(depth.device)
    depth = depth.contiguous().view(1, -1)
    pts = pts * depth
    pts = torch.inverse(K).mm(pts)
    return pts


def pts_to_image(pts, K, RT):
    wrld_X = RT[:3, :3].mm(pts) + RT[:3, 3:4]
    xy_proj = K.mm(wrld_X)
    EPS = 1e-2
    mask = (xy_proj[2:3, :].abs() < EPS).detach()
    zs = xy_proj[2:3, :]
    zs[mask] = EPS
    sampler = torch.cat((xy_proj[0:2, :] / zs, xy_proj[2:3, :]), 0)

    # Remove invalid zs that cause nans
    sampler[mask.repeat(3, 1)] = -10
    return sampler, wrld_X


def Screen_to_NDC(x, H, W):
    sampler = torch.clone(x)
    sampler[0:1] = (sampler[0:1] / (W -1) * 2.0 -1.0) * (W - 1.0) / (H - 1.0)
    sampler[1:2] = (sampler[1:2] / (H -1) * 2.0 -1.0)
    return sampler


def get_camera(world_to_cam, fov_in_degrees):
    # pytorch3d expects transforms as row-vectors, so flip rotation: https://github.com/facebookresearch/pytorch3d/issues/1183
    R = world_to_cam[:3, :3].t()[None, ...]
    T = world_to_cam[:3, 3][None, ...]
    camera = FoVPerspectiveCameras(device=world_to_cam.device, R=R, T=T, fov=fov_in_degrees, degrees=True)
    #K = get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees).to(world_to_cam.device)[None]
    #cameras = PerspectiveCameras(device=world_to_cam.device, R=R, T=T, in_ndc=False, K=K, image_size=torch.tensor([[H,W]]))
    return camera


def erp2sph(erp_points, erp_image_height=None, sph_modulo=False):
    # 0) the ERP image size
    if erp_image_height == None:
        height = np.shape(erp_points)[1]
        width = np.shape(erp_points)[2]
    else:
        height = erp_image_height
        width = height * 2

    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]

    # 1) point location to theta and phi
    points_theta = erp_points_x * (2 * np.pi / width) + np.pi / width - np.pi
    points_phi = -(erp_points_y * (np.pi / height) + np.pi / height * 0.5) + 0.5 * np.pi

    points_theta = np.where(points_theta == np.pi,  -np.pi, points_theta)
    points_phi = np.where(points_phi == -0.5 * np.pi, 0.5 * np.pi, points_phi)

    return np.stack((-points_theta, points_phi))


def car2sph(points_car, min_radius=1e-10):
    radius = np.linalg.norm(points_car, axis=1)

    valid_list = radius > min_radius  # set the 0 radius to origin.

    theta = np.zeros((points_car.shape[0]), float)
    theta[valid_list] = np.arctan2(points_car[:, 0][valid_list], points_car[:, 2][valid_list])

    phi = np.zeros((points_car.shape[0]), float)
    phi[valid_list] = -np.arcsin(np.divide(points_car[:, 1][valid_list], radius[valid_list]))

    return np.stack((theta, phi), axis=1)

def sph2erp(theta, phi, erp_image_height, sph_modulo=False):

    erp_image_width = 2 * erp_image_height
    erp_x = (theta + np.pi) / (2.0 * np.pi / erp_image_width) - 0.5
    erp_y = (-phi + 0.5 * np.pi) / (np.pi / erp_image_height) - 0.5
    return erp_x, erp_y


def sph2car(theta, phi, radius=1.0):
    # points_cartesian_3d = np.array.zeros((theta.shape[0],3),float)
    x = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.sin(phi)

    return np.stack((x, y, z), axis=0)


def erp2world(H, W, depth):
    xy_depth = get_pixel_grids(H, W, reverse=False).numpy()[:-1]
    theta, phi = erp2sph(xy_depth, H)
    world = sph2car(theta, phi, radius=depth.cpu().flatten().numpy())
    return torch.from_numpy(world).to(depth.device)


def unproject_points(world_to_cam, fov_in_degrees, depth, H, W):
    if (W == H*2):
        return erp2world(H, W, depth)
    camera = get_camera(world_to_cam, fov_in_degrees)

    xy_depth = get_pixel_grids(H, W, reverse=True).to(depth.device)
    xy_depth = Screen_to_NDC(xy_depth, H, W)
    xy_depth[2] = depth.flatten()
    xy_depth = xy_depth.T
    xy_depth = xy_depth[None]

    world_points = camera.unproject_points(xy_depth, world_coordinates=True, scaled_depth_input=False)
    #world_points = cameras.unproject_points(xy_depth, world_coordinates=True)
    world_points = world_points[0]
    world_points = world_points.T

    return world_points


def get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees=55.0):
    px, py = (W - 1) / 2., (H - 1) / 2.
    fx = fy = W / (2. * np.tan(fov_in_degrees / 360. * np.pi))
    k_ref = np.array([[fx, 0.0, px, 0.0], [0.0, fy, py, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=np.float32)
    k_ref = torch.tensor(k_ref)  # K is [4,4]

    return k_ref


def get_intrinsics(img, fov_in_degrees=55.0):
    C, H, W = img.shape
    if C != 3:
        H, W, C = img.shape
    k_ref = get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees)
    if isinstance(img, torch.Tensor):
        k_ref = k_ref.to(img.device)

    return k_ref


def get_extrinsics(rot_xyz, trans_xyz, device="cpu"):
    T = torch.tensor(trans_xyz)
    R = euler_angles_to_matrix(torch.tensor(rot_xyz), "XYZ")

    RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
    RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)

    return RT


def torch_to_o3d_mesh(vertices, faces, colors):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.T.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return mesh


def o3d_mesh_to_torch(mesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.triangles)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).T
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def torch_to_o3d_pcd(vertices, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return pcd


def o3d_pcd_to_torch(pcd, p=None, c=None):
    points = torch.from_numpy(np.asarray(pcd.points)).T
    if p is not None:
        points = points.to(p)
    colors = torch.from_numpy(np.asarray(pcd.colors)).T
    if c is not None:
        colors = colors.to(c)
    return points, colors


def torch_to_trimesh(vertices, faces, colors):
    mesh = trimesh.base.Trimesh(
        vertices=vertices.T.cpu().numpy(),
        faces=faces.T.cpu().numpy(),
        vertex_colors=(colors.T.cpu().numpy() * 255).astype(np.uint8),
        process=False)

    return mesh


def trimesh_to_torch(mesh: trimesh.base.Trimesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.faces)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors, dtype=float) / 255).T[:3]
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.base.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors).clip(0, 1) * 255).astype(np.uint8),
        process=False)
