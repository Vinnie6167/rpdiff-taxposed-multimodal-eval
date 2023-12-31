from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl
from equivariant_pose_graph.dataset.point_cloud_dataset import PointCloudDataset
from equivariant_pose_graph.dataset.point_cloud_dataset_test import TestPointCloudDataset


class MultiviewDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root='/home/bokorn/src/ndf_robot/notebooks',
                 test_dataset_root='/home/exx/Documents/ndf_robot/test_data/renders',
                 dataset_index=10,
                 action_class=0,
                 anchor_class=1,
                 dataset_size=1000,
                 rotation_variance=np.pi/180 * 5,
                 translation_variance=0.1,
                 batch_size=8,
                 num_workers=8,
                 cloud_type='final',
                 symmetric_class=None,
                 num_points=1024,
                 overfit=False,
                 num_overfit_transforms=3,
                 seed_overfit_transforms=False,
                 set_Y_transform_to_identity=False,
                 set_Y_transform_to_overfit=False,
                 gripper_lr_label=False,
                 no_transform_applied=False,
                 init_distribution_tranform_file='',
                 synthetic_occlusion=False,
                 ball_radius=None,
                 plane_occlusion=False,
                 ball_occlusion=False,
                 plane_standoff=None,
                 distractor_anchor_aug=False,
                 num_demo=12,
                 occlusion_class=0,
                 demo_mod_k_range=[2, 2],
                 demo_mod_rot_var=np.pi/180 * 360,
                 demo_mod_trans_var=0.15,
                 multimodal_transform_base=False,
                 rot_sample_method="axis_angle"
                 ):
        super().__init__()
        self.dataset_root = dataset_root
        self.test_dataset_root = test_dataset_root
        if isinstance(dataset_index, list):
            self.dataset_index = dataset_index
        elif dataset_index == None:
            self.dataset_index = None
        self.dataset_index = dataset_index
        self.no_transform_applied = no_transform_applied

        self.action_class = action_class
        self.anchor_class = anchor_class
        self.dataset_size = dataset_size
        self.rotation_variance = rotation_variance
        self.translation_variance = translation_variance
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cloud_type = cloud_type
        self.symmetric_class = symmetric_class
        self.num_points = num_points
        self.overfit = overfit
        self.num_overfit_transforms = num_overfit_transforms
        self.seed_overfit_transforms = seed_overfit_transforms
        # identity has a higher priority than overfit
        self.set_Y_transform_to_identity = set_Y_transform_to_identity
        self.set_Y_transform_to_overfit = set_Y_transform_to_overfit
        if self.set_Y_transform_to_identity:
            self.set_Y_transform_to_overfit = True
        self.gripper_lr_label = gripper_lr_label
        self.index_list = []
        self.init_distribution_tranform_file = init_distribution_tranform_file
        self.synthetic_occlusion = synthetic_occlusion
        self.ball_radius = ball_radius
        self.plane_occlusion = plane_occlusion
        self.ball_occlusion = ball_occlusion
        self.plane_standoff = plane_standoff
        self.distractor_anchor_aug = distractor_anchor_aug
        self.num_demo = num_demo
        self.occlusion_class = occlusion_class

        self.demo_mod_k_range = demo_mod_k_range
        self.demo_mod_rot_var = demo_mod_rot_var
        self.demo_mod_trans_var = demo_mod_trans_var
        self.multimodal_transform_base = multimodal_transform_base
        self.rot_sample_method = rot_sample_method

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        pass

    def setup(self, stage=None):
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            print("TRAIN Dataset")
            print(self.dataset_root)
            self.train_dataset = PointCloudDataset(
                dataset_root=self.dataset_root,
                dataset_indices=self.dataset_index,  # [self.dataset_index],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
                dataset_size=self.dataset_size,
                rotation_variance=self.rotation_variance,
                translation_variance=self.translation_variance,
                cloud_type=self.cloud_type,
                symmetric_class=self.symmetric_class,
                num_points=self.num_points,
                overfit=self.overfit,
                num_overfit_transforms=self.num_overfit_transforms,
                seed_overfit_transforms=self.seed_overfit_transforms,
                set_Y_transform_to_identity=self.set_Y_transform_to_identity,
                set_Y_transform_to_overfit=self.set_Y_transform_to_overfit,
                gripper_lr_label=self.gripper_lr_label,
                synthetic_occlusion=self.synthetic_occlusion,
                ball_radius=self.ball_radius,
                plane_occlusion=self.plane_occlusion,
                ball_occlusion=self.ball_occlusion,
                plane_standoff=self.plane_standoff,
                distractor_anchor_aug=self.distractor_anchor_aug,
                num_demo=self.num_demo,
                occlusion_class=self.occlusion_class,
                demo_mod_k_range=self.demo_mod_k_range,
                demo_mod_rot_var=self.demo_mod_rot_var,
                demo_mod_trans_var=self.demo_mod_trans_var,
                multimodal_transform_base=self.multimodal_transform_base,
                rot_sample_method=self.rot_sample_method,
                random_files=True,
            )

        if stage == 'val' or stage is None:
            print("VAL Dataset")
            self.val_dataset = PointCloudDataset(
                dataset_root=self.test_dataset_root,
                dataset_indices=self.dataset_index,  # [self.dataset_index],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
                dataset_size=self.dataset_size,
                rotation_variance=self.rotation_variance,
                translation_variance=self.translation_variance,
                cloud_type=self.cloud_type,
                symmetric_class=self.symmetric_class,
                num_points=self.num_points,
                overfit=self.overfit,
                num_overfit_transforms=self.num_overfit_transforms,
                seed_overfit_transforms=self.seed_overfit_transforms,
                set_Y_transform_to_identity=self.set_Y_transform_to_identity,
                set_Y_transform_to_overfit=self.set_Y_transform_to_overfit,
                gripper_lr_label=self.gripper_lr_label,
                synthetic_occlusion=self.synthetic_occlusion,
                ball_radius=self.ball_radius,
                plane_occlusion=self.plane_occlusion,
                ball_occlusion=self.ball_occlusion,
                plane_standoff=self.plane_standoff,
                distractor_anchor_aug=self.distractor_anchor_aug,
                num_demo=None,
                occlusion_class=self.occlusion_class,
                demo_mod_k_range=self.demo_mod_k_range,
                demo_mod_rot_var=self.demo_mod_rot_var,
                demo_mod_trans_var=self.demo_mod_trans_var,
                multimodal_transform_base=self.multimodal_transform_base,
                rot_sample_method=self.rot_sample_method,
                random_files=False,
            )
        if stage == 'test':
            print("TEST Dataset")
            self.test_dataset = TestPointCloudDataset(
                dataset_root=self.test_dataset_root,
                dataset_indices=self.dataset_index,  # [self.dataset_index],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
                dataset_size=self.dataset_size,
                rotation_variance=self.rotation_variance,
                translation_variance=self.translation_variance,
                cloud_type=self.cloud_type,
                symmetric_class=self.symmetric_class,
                num_points=self.num_points,
                overfit=self.overfit,
                num_overfit_transforms=self.num_overfit_transforms,
                seed_overfit_transforms=self.seed_overfit_transforms,
                set_Y_transform_to_identity=self.set_Y_transform_to_identity,
                set_Y_transform_to_overfit=self.set_Y_transform_to_overfit,
                gripper_lr_label=self.gripper_lr_label,
                index_list=self.index_list,
                no_transform_applied=self.no_transform_applied,
                init_distribution_tranform_file=self.init_distribution_tranform_file,
                demo_mod_k_range=self.demo_mod_k_range,
                demo_mod_rot_var=self.demo_mod_rot_var,
                demo_mod_trans_var=self.demo_mod_trans_var,
                random_files=False
            )

    def return_index_list_test(self):
        return self.test_dataset.return_index_list()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
