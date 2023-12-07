from torch.utils.data import DataLoader
import numpy as np
import torch
from equivariant_pose_graph.utils.se3 import random_se3
from pytorch3d.transforms import Rotate
from torch.utils.data import Dataset

import pytorch_lightning as pl

from rpdiff.training import dataio_full_chunked as dataio
import os.path as osp
from torch.utils.data import DataLoader
from rpdiff.utils import config_util, path_util

class RPDiffDatasetWrapper(Dataset):
    def __init__(self, 
                 original_dataset,
                 rotation_variance=np.pi,
                 translation_variance=0.5,
                 overfit=False,
                 num_overfit_transforms=3,
                 seed_overfit_transforms=False,
                 set_Y_transform_to_identity=False,
                 set_Y_transform_to_overfit=False,
                 rot_sample_method="axis_angle"
                 ):
        self.original_dataset = original_dataset

        self.rot_var = rotation_variance
        self.trans_var = translation_variance

        self.overfit = overfit
        self.seed_overfit_transforms = seed_overfit_transforms
        # identity has a higher priority than overfit
        self.set_Y_transform_to_identity = set_Y_transform_to_identity
        self.set_Y_transform_to_overfit = set_Y_transform_to_overfit
        if self.set_Y_transform_to_identity:
            self.set_Y_transform_to_overfit = True
        self.num_overfit_transforms = num_overfit_transforms
        self.T0_list = []
        self.T1_list = []
        self.T2_list = []

        self.rot_sample_method = rot_sample_method

        if self.overfit or self.set_Y_transform_to_overfit:
            self.get_fixed_transforms()

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]

        # Parse out the demonstration data
        points_anchor = torch.tensor(sample[1][1]['parent_final_pcd'])[None,].float()
        points_action = torch.tensor(sample[1][1]['child_final_pcd'])[None,].float()

        ########
        # The below is transformation code copied from TAXPose's PointCloudDataset

        if self.overfit:
            transform_idx = torch.randint(
                self.num_overfit_transforms, (1,)).item()
            T0 = self.T0_list[transform_idx]
            T1 = self.T1_list[transform_idx]
            if not self.set_Y_transform_to_overfit:
                T2 = random_se3(1, rot_var=self.rot_var,
                                trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)
            else:
                T2 = self.T2_list[transform_idx]
        else:
            T0 = random_se3(1, rot_var=self.rot_var,
                            trans_var=self.trans_var, device=points_action.device, rot_sample_method=self.rot_sample_method)
            T1 = random_se3(1, rot_var=self.rot_var,
                            trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)
            if self.set_Y_transform_to_identity:
                T2 = Rotate(torch.eye(3), device=points_anchor.device)
            elif self.set_Y_transform_to_overfit:
                transform_idx = torch.randint(
                        self.num_overfit_transforms, (1,)).item()
                T2 = self.T2_list[transform_idx]
            else:
                T2 = random_se3(1, rot_var=self.rot_var,
                                trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)

        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)

        points_action_onetrans = T2.transform_points(points_action)
        points_anchor_onetrans = T2.transform_points(points_anchor)

        data = {
            'points_action': points_action.squeeze(0),
            'points_anchor': points_anchor.squeeze(0),
            'points_action_trans': points_action_trans.squeeze(0),
            'points_anchor_trans': points_anchor_trans.squeeze(0),
            'points_action_onetrans': points_action_onetrans.squeeze(0),
            'points_anchor_onetrans': points_anchor_onetrans.squeeze(0),
            'T0': T0.get_matrix().squeeze(0),
            'T1': T1.get_matrix().squeeze(0),
            'T2': T2.get_matrix().squeeze(0),
            'symmetric_cls': torch.tensor([]),
            # 'mug_id': filename.name.split("_")[0],
        }

        return data

class RpDiffDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=16,
                 ):
        super().__init__()
        self.batch_size = batch_size

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        pass

    def setup(self, stage=None):
        config_fname = "mug_on_rack_multi_cfgs/mug_on_rack_multi_pose_diff_with_varying_crop_fixed_noise_var.yaml"
        train_args = config_util.load_config(osp.join(path_util.get_train_config_dir(), config_fname))
        
        # train_args['debug'] = args.debug
        # train_args['debug_data'] = args.debug_data
        # train_args['port_vis'] = args.port_vis
        # train_args['seed'] = args.seed
        # train_args['local_dataset_dir'] = args.local_dataset_dir
        # train_args['meshcat_ap'] = args.meshcat
        # train_args['resume_ap'] = args.resume

        train_args = config_util.recursive_attr_dict(train_args)
        data_args = train_args.data

        # don't add any noise to the demonstrations
        data_args.refine_pose.diffusion_steps = 0

        # data_args = args.data
        dataset_path = osp.join(
            path_util.get_rpdiff_data(), 
            data_args.data_root,
            data_args.dataset_path)

        assert osp.exists(dataset_path), f'Dataset path: {dataset_path} does not exist'

        self.train_dataset = dataio.FullRelationPointcloudPolicyDataset(
            dataset_path, 
            data_args,
            phase='train', 
            train_coarse_aff=False,
            train_refine_pose=True,
            train_success=False,
            mc_vis=None, 
            debug_viz=False) #args.debug_data)
        self.val_dataset = dataio.FullRelationPointcloudPolicyDataset(
            dataset_path, 
            data_args,
            phase='val', 
            train_coarse_aff=False,
            train_refine_pose=True,
            train_success=False,
            mc_vis=None,
            debug_viz=False) #args.debug_data)

    def return_index_list_test(self):
        return self.test_dataset.return_index_list()

    def train_dataloader(self):
        return DataLoader(
            RPDiffDatasetWrapper(self.train_dataset, rot_sample_method="axis_angle"), 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=12, 
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            RPDiffDatasetWrapper(self.val_dataset, rot_sample_method="axis_angle"), 
            batch_size=2, 
            num_workers=1,
            shuffle=False, 
            drop_last=True)

    def test_dataloader(self):
        return DataLoader(RPDiffDatasetWrapper(self.test_dataset, rot_sample_method="axis_angle"),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
