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
import os.path as osp


def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds


class ThreeDFrontDataset(Data.Dataset):
    def __init__(self,
                 split='train',
                 config=None,
                 ):
        self.config = config
        self.root = config.data.root
        self.split = split
        if not split=='train':
            file_number = config.data.test_file_number
        else:
            file_number = config.data.train_file_number

        self.files = []
        self.length = 0
        self.max_points = 15000
        if split == 'test':
            print('construct test dataset')
            self.data_list = self._build_data_list(config.data.test_name, file_number[0])
            return
        if split == 'val':
            print('construct val dataset')
            self.data_list = self._build_data_list('test/sp/high', file_number[0])
            self.data_list.extend(self._build_data_list('test/sp/low', file_number[1]))
            self.data_list.extend(self._build_data_list('test/bp/high', file_number[2]))
            self.data_list.extend(self._build_data_list('test/bp/low', file_number[3]))
        else:
            print('construct train dataset')
            self.data_list = self._build_data_list('rawdata/sp/high', file_number[0])
            self.data_list.extend(self._build_data_list('rawdata/sp/low', file_number[1]))
            self.data_list.extend(self._build_data_list('rawdata/bp/high', file_number[2]))
            self.data_list.extend(self._build_data_list('rawdata/bp/low', file_number[3]))

    def _build_data_list(self,file_name='rawdata/sp/high',file_number=1000):
        data_list = []
        subset_path = osp.join(self.root, file_name)

        total = 0
        scene_ids = os.listdir(subset_path)

        for scene_id in scene_ids:
            scene_path = osp.join(subset_path, scene_id)
            if osp.isdir(scene_path):
                data_list.append(osp.join(file_name, scene_id))
                total += 1
                if total >= file_number:
                    break
        return data_list


    def __getitem__(self, index):
        scene_id = self.data_list[index]
        scene_path = osp.join(self.root , scene_id)
        if self.config.data.wo_anim:
            ref_points = np.load(osp.join(scene_path, 'ref_wo_anim.npy'))
        else:
            ref_points = np.load(osp.join(scene_path, 'ref.npy'))
        src_points = np.load(osp.join(scene_path, 'src.npy'))
        transform = np.load(osp.join(scene_path, 'relative_transform.npy'))
        

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_points)
        src_pcd.paint_uniform_color([1, 0.706, 0])
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(ref_points)
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=self.config.data.downsample)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        if self.split == 'train':            
            src_pts += (np.random.rand(src_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            tgt_pts += (np.random.rand(tgt_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise

        # second sample
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt) 

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt) 

        # if we get too many points, we do some downsampling
        if (src_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            src_kpt = src_kpt[idx]

        if (tgt_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            tgt_kpt = tgt_kpt[idx]

        if self.split == 'test':
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
                'relt_pose': transform,
                'src_sds_pts': src_kpt, # second downsampling
                'tgt_sds_pts': tgt_kpt}


    def __len__(self):
        return len(self.data_list)
