import torch.utils.data as Data
import os
from os.path import join, exists
import pickle
import open3d as o3d
import glob
from utils.SE3 import *
from utils.tools import get_pcd, get_keypts, loadlog
from utils.common import make_open3d_point_cloud
import copy
import gc


def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds


class ETHTestset(Data.Dataset):
    def __init__(self,
                 config=None
                 ):
        self.config = config
        self.root = config.data.root
        self.files = []
        self.length = 0

        scene_list = [
            'gazebo_summer',
            'gazebo_winter',
            'wood_autmn',
            'wood_summer',
        ]
        self.poses = []
        for scene in scene_list:
            pcdpath = self.root + f'/{scene}/'
            gtpath = self.root + f'/{scene}/'
            gtLog = loadlog(gtpath)
            for key in gtLog.keys():
                id1 = key.split('_')[0]
                id2 = key.split('_')[1]
                src_id = join(pcdpath, f'Hokuyo_{id1}')
                tgt_id = join(pcdpath, f'Hokuyo_{id2}')
                self.files.append([src_id, tgt_id])
                self.length += 1
                self.poses.append(gtLog[key])

    def __getitem__(self, index):

        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]

        # load src fragment
        src_path = os.path.join(self.root, src_id)
        src_pcd = o3d.io.read_point_cloud(src_path + '.ply')
        src_pcd.paint_uniform_color([1, 0.706, 0])
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        # load tgt fragment
        tgt_path = os.path.join(self.root, tgt_id)
        tgt_pcd = o3d.io.read_point_cloud(tgt_path + '.ply')
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=self.config.data.downsample)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        # relative pose
        relt_pose = np.linalg.inv(self.poses[index])

        # voxel sampling
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

        # if we get too many points, we do random downsampling
        if (src_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            src_kpt = src_kpt[idx]

        if (tgt_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            tgt_kpt = tgt_kpt[idx]

        src_pcd = make_open3d_point_cloud(src_kpt)
        src_pcd.estimate_normals()
        src_pcd.orient_normals_towards_camera_location()
        src_noms = np.array(src_pcd.normals)
        src_kpt = np.concatenate([src_kpt, src_noms], axis=-1)

        tgt_pcd = make_open3d_point_cloud(tgt_kpt)
        tgt_pcd.estimate_normals()
        tgt_pcd.orient_normals_towards_camera_location()
        tgt_noms = np.array(tgt_pcd.normals)
        tgt_kpt = np.concatenate([tgt_kpt, tgt_noms], axis=-1)


        return {'src_fds_pts': src_pts, # first downsampling
                'tgt_fds_pts': tgt_pts,
                'relt_pose': relt_pose,
                'src_sds_pts': src_kpt, # second downsampling
                'tgt_sds_pts': tgt_kpt,
                'src_id': src_id,
                'tgt_id': tgt_id}


    def __len__(self):
        return self.length
