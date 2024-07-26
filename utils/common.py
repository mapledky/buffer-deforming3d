import open3d
import numpy as np
import os
import time
import torch
from sklearn.neighbors import KDTree
import pointnet2_ops.pointnet2_utils as pnt2
import torch.nn.functional as F
from torch.autograd import Variable
from torch_batch_svd import svd
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy
from utils.SE3 import *


def plot(x, text=None, normalize=False):
    assert x.size(0) == 1
    assert x.size(1) in [1, 3]
    x = x[0]
    if x.dim() == 4:
        x = x[..., 0]

    nch = x.size(0)
    is_rgb = (nch == 3)

    if normalize:
        x = x - x.view(nch, -1).mean(-1).view(nch, 1, 1)
        x = 0.4 * x / x.view(nch, -1).std(-1).view(nch, 1, 1)

    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0)).clip(0, 1)

    print(x.shape)
    if is_rgb:
        plt.imshow(x)
    else:
        plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis("off")

    if text is not None:
        plt.text(0.5, 0.5, text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 color='white', fontsize=150)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def select_patches(pts, refer_pts, vicinity, patch_sample=1024):
    B, N, C = pts.shape

    # shuffle pts if pts is not orderless
    index = np.random.choice(N, N, replace=False)
    pts = pts[:, index]

    group_idx = pnt2.ball_query(vicinity, patch_sample, pts, refer_pts)
    pts_trans = pts.transpose(1, 2).contiguous()
    new_points = pnt2.grouping_operation(
        pts_trans, group_idx
    )
    new_points = new_points.permute([0, 2, 3, 1])
    mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, patch_sample)
    mask = (group_idx == mask).float()
    mask[:, :, 0] = 0
    mask[:, :, patch_sample - 1] = 1
    mask = mask.unsqueeze(3).repeat([1, 1, 1, C])
    new_pts = refer_pts.unsqueeze(2).repeat([1, 1, patch_sample, 1])
    local_patches = new_points * (1 - mask).float() + new_pts * mask.float()

    del mask
    del new_points
    del group_idx
    del new_pts
    del pts
    del pts_trans

    return local_patches


def transform_pc_pytorch(pc, sn):
    '''

    :param pc: 3xN tensor
    :param sn: 5xN tensor / 4xN tensor
    :param node: 3xM tensor
    :return: pc, sn, node of the same shape, detach
    '''
    angles_3d = np.random.rand(3) * np.pi * 2
    shift = np.random.uniform(-1, 1, (1, 3))

    sigma, clip = 0.010, 0.02
    N, C = pc.shape
    jitter_pc = np.clip(sigma * np.random.randn(N, 3), -1 * clip, clip)
    sigma, clip = 0.010, 0.02
    jitter_sn = np.clip(sigma * np.random.randn(N, 4), -1 * clip, clip)
    pc += jitter_pc
    sn += jitter_sn

    pc = pc_rotate_translate(pc, angles_3d, shift)
    sn[:, 0:3] = vec_rotate(sn[:, 0:3], angles_3d)  # 3x3 * 3xN -> 3xN

    return pc, sn, angles_3d, shift


def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def pc_rotate_translate(data, angles, translates):
    '''
    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :param translates: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, np.transpose(R)) + translates

    return rotated_data


def pc_rotate_translate_torch(data, angles, translates=None):
    '''
    :param data: Tensor of BxNx3 array
    :param angles: Tensor of Bx3
    :param translates: Tensor of Bx3
    :return: rotated_data: Tensor of Nx3
    '''
    device = data.device
    B, N, _ = data.shape

    R = np.zeros([B, 3, 3])
    for i in range(B):
        R[i] = angles2rotation_matrix(angles[i])  # 3x3
    R = torch.FloatTensor(R).to(device)

    if not translates is None:
        rotated_data = torch.matmul(data, R.transpose(-1, -2)) + torch.FloatTensor(translates).unsqueeze(1).to(device)
        return rotated_data
    else:
        rotated_data = torch.matmul(data, R.transpose(-1, -2))
        return rotated_data


def max_ind(data):
    B, C, row, col = data.shape
    inds = np.zeros([B, 2])
    for i in range(B):
        ind = torch.argmax(data[i])
        r = int(ind // col)
        c = ind % col
        inds[i, 0] = r
        inds[i, 1] = c
    return inds


def vec_rotate(data, angles):
    '''
    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data


def vec_rotate_torch(data, angles):
    '''
    :param data: BxNx3 tensor
    :param angles: Bx3 numpy array
    :return:
    '''
    device = data.device
    B, N, _ = data.shape

    R = np.zeros([B, 3, 3])
    for i in range(B):
        R[i] = angles2rotation_matrix(angles[i])  # 3x3
    R = torch.FloatTensor(R).to(device)

    rotated_data = torch.matmul(data, R.transpose(-1, -2))  # BxNx3 * Bx3x3 -> BxNx3
    return rotated_data


def rotate_perturbation_point_cloud(data, angle_sigma=0.01, angle_clip=0.05):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # truncated Gaussian sampling
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    rotated_data = vec_rotate(data, angles)

    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original point clouds
        Return:
          BxNx3 array, jittered point clouds
    """
    B, N, C = data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += data
    return jittered_data


def cdist(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    diff = a.unsqueeze(0) - b.unsqueeze(1)
    dis_matrix = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    return dis_matrix


def s2_grid(n_alpha, n_beta):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    # beta = np.linspace(start=0, stop=np.pi, num=n_beta+2, endpoint=False) + np.pi / n_beta / 2
    # beta = beta[1:-1]
    beta = np.linspace(start=0, stop=np.pi, num=n_beta, endpoint=False) + np.pi / n_beta / 2

    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False) + np.pi / n_alpha
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return grid


def pad_image(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    if kernel_size % 2 == 0:
        pad_size = kernel_size // 2
        output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], pad_size, output.shape[3]]).to(device)
        output = torch.cat([output, zeros_pad], dim=2)
    else:
        pad_size = (kernel_size - 1) // 2
        output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
        output = torch.cat([input[:, :, :, -pad_size:], output], dim=3)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], pad_size, output.shape[3]]).to(device)
        output = torch.cat([output, zeros_pad], dim=2)
        output = torch.cat([zeros_pad, output], dim=2)
    return output


def pad_image_3d(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, D, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    if kernel_size % 2 == 0:
        pad_size = kernel_size // 2
        output = torch.cat([input, input[:, :, :, :, 0:pad_size]], dim=4)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], output.shape[2], pad_size, output.shape[4]]).to(
            device)
        output = torch.cat([output, zeros_pad], dim=3)
    else:
        pad_size = (kernel_size - 1) // 2
        output = torch.cat([input, input[:, :, :, :, 0:pad_size]], dim=4)
        output = torch.cat([input[:, :, :, :, -pad_size:], output], dim=4)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], output.shape[2], pad_size, output.shape[4]]).to(
            device)
        output = torch.cat([output, zeros_pad], dim=3)
        output = torch.cat([zeros_pad, output], dim=3)
    return output


def pad_image_3d_(input, *kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, D, H, W]
    :param kernel_size: (Depth, Height, Width)
    :return:
    """
    device = input.device
    D, H, W = kernel_size[0]

    pad_size = (W - 1) // 2
    if pad_size != 0:
        output = torch.cat([input, input[:, :, :, :, 0:pad_size]], dim=4)
        output = torch.cat([input[:, :, :, :, -pad_size:], output], dim=4)
    else:
        output = input

    pad_size = (H - 1) // 2
    if pad_size != 0:
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], output.shape[2], pad_size, output.shape[4]]).to(device)
        output = torch.cat([output, zeros_pad], dim=3)
        output = torch.cat([zeros_pad, output], dim=3)
    return output


def pad_3d_on_rad(input, *kernel_size):
    """
    Performing zeros padding on rad dimension for 3d image
    :param input: [B, C, D, H, W]
    :param kernel_size: (Depth, Height, Width)
    :return:
    """

    D, H, W = kernel_size[0]

    pad_size = (D - 1) // 2
    output = torch.cat([input, torch.zeros_like(input)[:, :, 0:pad_size]], dim=2)
    output = torch.cat([torch.zeros_like(input)[:, :, 0:pad_size], output], dim=2)
    return output


def pad_image_on_azi(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    pad_size = (kernel_size - 1) // 2
    output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
    output = torch.cat([input[:, :, :, -pad_size:], output], dim=3)
    return output


def kmax_pooling(x, dim, k):
    kmax = x.topk(k, dim=dim)[0]
    return kmax


def change_coordinates(coords, radius, p_from='C', p_to='S'):
    """
    Change Spherical to Cartesian coordinates and vice versa, for points x in S^2.

    In the spherical system, we have coordinates beta and alpha,
    where beta in [0, pi] and alpha in [0, 2pi]

    We use the names beta and alpha for compatibility with the SO(3) code (S^2 being a quotient SO(3)/SO(2)).
    Many sources, like wikipedia use theta=beta and phi=alpha.

    :param coords: coordinate array
    :param p_from: 'C' for Cartesian or 'S' for spherical coordinates
    :param p_to: 'C' for Cartesian or 'S' for spherical coordinates
    :return: new coordinates
    """
    if p_from == p_to:
        return coords
    elif p_from == 'S' and p_to == 'C':

        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = radius

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct  # z
        return out

    elif p_from == 'C' and p_to == 'S':

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        out = np.empty(x.shape + (2,))
        out[..., 0] = np.arccos(z)  # beta
        out[..., 1] = np.arctan2(y, x)  # alpha
        return out

    else:
        raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))


def get_voxel_coordinate(radius, rad_n, azi_n, ele_n):
    grid = s2_grid(n_alpha=azi_n, n_beta=ele_n)
    pts_xyz_on_S2 = change_coordinates(grid, radius, 'S', 'C')
    pts_xyz_on_S2 = np.expand_dims(pts_xyz_on_S2, axis=0).repeat(rad_n, axis=0)
    scale = np.reshape(np.arange(rad_n) / rad_n + 1 / (2 * rad_n), [rad_n, 1, 1])
    pts_xyz = scale * pts_xyz_on_S2
    return pts_xyz


def sphere_query(pts, new_pts, radius, nsample):
    """
    :param pts: all points, [B. N. C]
    :param new_pts: query points, [B, S. 3]
    :param radius: local sperical radius
    :param nsample: max sample number in local sphere
    :return:
    """
    B, N, C = pts.shape
    pts = pts.contiguous()
    new_pts = new_pts.contiguous()
    group_idx = pnt2.ball_query(radius, nsample, pts[:, :, :3].contiguous(), new_pts[:, :, :3].contiguous())
    mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample)
    mask = (group_idx == mask).float()
    mask[:, :, 0] = 0

    mask1 = (group_idx[:,:,0] == 0).unsqueeze(2).float()
    mask1 = torch.cat([mask1, torch.zeros_like(mask)[:,:,:-1]], dim=2)
    mask = mask + mask1

    # C implementation
    pts_trans = pts.transpose(1, 2).contiguous()
    new_points = pnt2.grouping_operation(
        pts_trans, group_idx
    )  # (B, 3, npoint, nsample)
    new_points = new_points.permute([0, 2, 3, 1])

    # replace the wrong points using new_pts
    mask = mask.unsqueeze(3).repeat([1, 1, 1, C])
    n_points = new_points*(1-mask).float()

    del mask
    del new_points
    del group_idx
    del new_pts
    del pts
    del pts_trans

    return n_points


def var_to_invar(pts, rad_n, azi_n, ele_n):
    """
    :param pts: input points data, [B, N, nsample, 3]
    :param rad_n: radial number
    :param azi_n: azimuth number
    :param ele_n: elevator number
    :return:
    """
    device = pts.device
    B, N, nsample, C = pts.shape
    assert N == rad_n * azi_n * ele_n
    angle_step = np.array([0, 0, 2 * np.pi / azi_n])
    pts = pts.view(B, rad_n, ele_n, azi_n, nsample, C)

    R = np.zeros([azi_n, 3, 3])
    for i in range(azi_n):
        angle = -1 * i * angle_step
        r = angles2rotation_matrix(angle)
        R[i] = r
    R = torch.FloatTensor(R).to(device)
    R = R.view(1, 1, 1, azi_n, 3, 3).repeat(B, rad_n, ele_n, 1, 1, 1)
    new_pts = torch.matmul(pts, R.transpose(-1, -2))

    del R
    del pts

    return new_pts.view(B, -1, nsample, C)


def RodsRotatFormula(a, b):
    B, _ = a.shape
    device = a.device
    b = b.to(device)
    c = torch.cross(a, b)
    theta = torch.acos(F.cosine_similarity(a, b)).unsqueeze(1).unsqueeze(2)

    c = F.normalize(c, p=2, dim=1)
    one = torch.ones(B, 1, 1).to(device)
    zero = torch.zeros(B, 1, 1).to(device)
    a11 = zero
    a12 = -c[:, 2].unsqueeze(1).unsqueeze(2)
    a13 = c[:, 1].unsqueeze(1).unsqueeze(2)
    a21 = c[:, 2].unsqueeze(1).unsqueeze(2)
    a22 = zero
    a23 = -c[:, 0].unsqueeze(1).unsqueeze(2)
    a31 = -c[:, 1].unsqueeze(1).unsqueeze(2)
    a32 = c[:, 0].unsqueeze(1).unsqueeze(2)
    a33 = zero
    Rx = torch.cat(
        (torch.cat((a11, a12, a13), dim=2), torch.cat((a21, a22, a23), dim=2), torch.cat((a31, a32, a33), dim=2)),
        dim=1)
    I = torch.eye(3).to(device)
    R = I.unsqueeze(0).repeat(B, 1, 1) + torch.sin(theta) * Rx + (1 - torch.cos(theta)) * torch.matmul(Rx, Rx)
    return R.transpose(-1, -2)


def vec_to_vec_rot_matrix(a, b):
    '''

    :param a: source vector [B, 3] Tensor
    :param b: target vector [B, 3] Tensor
    :return: rotation matrix R [B, 3, 3] Tensor
    '''
    B, _ = a.shape
    device = a.device
    b = b.to(device)
    c = torch.cross(a, b)
    theta = torch.acos(F.cosine_similarity(a, b)).unsqueeze(1).unsqueeze(2)

    rotate_axis = torch.cross(a, b)
    rotate_axis = F.normalize(rotate_axis, p=2, dim=1)
    third_axis = torch.cross(a, rotate_axis)
    third_axis = F.normalize(third_axis, p=2, dim=1)
    LRF = torch.cat((rotate_axis.unsqueeze(2), third_axis.unsqueeze(2), a.unsqueeze(2)), dim=2)
    trans_cord = torch.matmul(LRF.transpose(-1, -2), b.unsqueeze(2))
    # sin = trans_cord[:, 1, :]
    # cos = trans_cord[:, 2, :]
    one = torch.ones(10, 1, 1).to(device)
    zero = torch.zeros(10, 1, 1).to(device)
    a11 = one
    a12 = zero
    a13 = zero
    a21 = zero
    a22 = torch.cos(theta)
    a23 = -torch.sin(theta)
    a31 = zero
    a32 = torch.sin(theta)
    a33 = torch.cos(theta)
    Rx = torch.cat(
        (torch.cat((a11, a12, a13), dim=2), torch.cat((a21, a22, a23), dim=2), torch.cat((a31, a32, a33), dim=2)),
        dim=1)
    # I = torch.eye(3).to(device)
    # R = I.unsqueeze(0).repeat(B,1,1) + torch.sin(theta)*Rx + (1-torch.cos(theta))*Rx*Rx

    return torch.matmul(LRF, Rx).transpose(-1, -2)


def make_open3d_point_cloud(xyz, color=None):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        pcd.colors = open3d.utility.Vector3dVector(color)
    return pcd


def mesh_sphere(pcd, voxel_size, sphere_size=0.5):
    pcd = open3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=voxel_size)
    # Create a mesh sphere
    spheres = open3d.geometry.TriangleMesh()
    s = open3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()

    for i, p in enumerate(pcd.points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def plot_corres(sraw, traw, skpts, tkpts, trans, thr, align=False):
    '''
    Args:
        sraw:     array  [P, 3]
        traw:     array  [Q, 3]
        skpts:    array  [N, 3]
        tkpts:    array  [N, 3]
        trans:    array  [4, 4]

    Returns:

    '''
    len = skpts.shape[0]
    t_skpts = transform(skpts, trans)
    mask = (np.sum((t_skpts - tkpts) ** 2, axis=-1) < thr ** 2)
    inlier_rate = mask.sum() / len
    colors = np.zeros((len, 3))
    colors[mask] = [0, 1, 0]
    colors[~mask] = [1, 0, 0]

    # visulization
    offset = 2
    offset = np.array([0, 0, offset])[None]

    if align is True:
        sraw = transform(sraw, trans)
        skpts = transform(skpts, trans)
    sraw_pcd = make_open3d_point_cloud(sraw, [227/255, 207/255, 87/255])
    # sraw_pcd = mesh_sphere(sraw_pcd, voxel_size=0.3)
    sraw_pcd.estimate_normals()
    traw_pcd = make_open3d_point_cloud(traw + offset, [0, 0.651, 0.929])
    # traw_pcd = mesh_sphere(traw_pcd, voxel_size=0.3)
    traw_pcd.estimate_normals()

    vertice = np.concatenate([skpts, tkpts + offset], axis=0)
    line = np.concatenate([np.arange(0, len)[:, None], np.arange(0, len)[:, None] + len], axis=-1)
    lines_pcd = plot_correspondences(vertice, line, colors)

    open3d.visualization.draw_geometries([sraw_pcd, traw_pcd, lines_pcd])
    return inlier_rate


def plot_correspondences(points, lines, color):
    '''
    Args:
        points:  initial point sets [2N, 3]
        lines:   indices of points  [N, 2]
        color:

    Returns:
    '''
    lines_pcd = open3d.geometry.LineSet()
    lines_pcd.lines = open3d.utility.Vector2iVector(lines)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], lines.shape[0], axis=0)
        lines_pcd.colors = open3d.utility.Vector3dVector(color)
    lines_pcd.points = open3d.utility.Vector3dVector(points)

    return lines_pcd

# convert data
def convert_data(data, color, ind=None):
    data = data.squeeze()
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    color = color * 0.5
    min = 0#np.array(data).min() #
    max = 1.0#np.array(data).max() #
    data[data > max] = max
    data[data < min] = min

    x = np.arange(0, 10, 1)
    y = np.arange(0, 10, 1)
    X, Y = np.meshgrid(x, y)
    Z = 2 * (np.sin(X) + np.sin(3 * Y))
    plt.pcolor(X, Y, Z, cmap=cm.rainbow, vmin=min, vmax=max)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.show()

    norm = matplotlib.colors.Normalize(vmin = min, vmax=max, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    rgba = mapper.to_rgba(data)

    if ind is None:
        color = rgba[:, :3]
    else:
        for i in range(len(ind)):
            value = ind[i]
            color[value] = rgba[i, :3]

    return color


def render_pc(pc, score):
    """
    render the input point cloud according to its point-wise score
    Args:
        pc:     Nx3 array, input xyz coordinates
        score:  N array

    Returns:
        pcd:    output colored point cloud
    """
    color = np.ones([score.shape[0], 3])
    color = convert_data(score, color, None)
    pcd = make_open3d_point_cloud(pc, color)
    return pcd


def cal_Z_axis(local_cor, local_weight=None, ref_point=None, disambigutiy='normal'):
    device = local_cor.device
    B, N, _ = local_cor.shape
    cov_matrix = torch.matmul(local_cor.transpose(-1, -2), local_cor) if local_weight is None \
        else Variable(torch.matmul(local_cor.transpose(-1, -2), local_cor * local_weight), requires_grad=True)
    # Z_axis = torch.symeig(cov_matrix, eigenvectors=True)[1][:, :, 0]
    u, s, v = svd(cov_matrix)
    Z_axis = u[:,:,-1]
    if disambigutiy == 'normal':
        mask = (torch.sum(-Z_axis * ref_point, dim=1) < 0).float().unsqueeze(1)
    else:
        temp = torch.zeros_like(Z_axis)
        temp[:, -1] = 1
        mask = (F.cosine_similarity(torch.sum(local_cor, dim=1), temp) < 0).float().unsqueeze(1)

    Z_axis = Z_axis * (1 - mask) - Z_axis * mask

    return Z_axis
