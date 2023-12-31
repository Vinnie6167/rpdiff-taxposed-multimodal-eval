import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch3d.transforms import Transform3d, Translate, matrix_to_axis_angle
from torchvision.transforms import ToTensor
from pytorch3d.loss import chamfer_distance
from equivariant_pose_graph.training.point_cloud_training_module import PointCloudTrainingModule
from equivariant_pose_graph.utils.se3 import dualflow2pose, flow2pose, get_translation, get_degree_angle, dense_flow_loss
from equivariant_pose_graph.utils.display_headless import scatter3d, quiver3d
from equivariant_pose_graph.utils.color_utils import get_color

import wandb

mse_criterion = nn.MSELoss(reduction='sum')
to_tensor = ToTensor()


class EquivarianceTrainingModule(PointCloudTrainingModule):

    def __init__(self,
                 model=None,
                 lr=1e-3,
                 image_log_period=500,
                 action_weight=1,
                 anchor_weight=1,
                 smoothness_weight=0.1,
                 consistency_weight=1,
                 rotation_weight=0,
                 chamfer_weight=10000,
                 point_loss_type=0,
                 return_flow_component=False,
                 weight_normalize='l1',
                 sigmoid_on=False,
                 softmax_temperature=None):
        super().__init__(model=model, lr=lr,
                         image_log_period=image_log_period,)
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.smoothness_weight = smoothness_weight
        self.rotation_weight = rotation_weight
        self.chamfer_weight = chamfer_weight
        self.consistency_weight = consistency_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize
        # 0 for mse loss, 1 for chamfer distance, 2 for mse loss + chamfer distance
        self.point_loss_type = point_loss_type
        self.return_flow_component = return_flow_component
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        if self.weight_normalize == 'l1':
            assert (self.sigmoid_on), "l1 weight normalization need sigmoid on"

    def compute_loss(self, x_action, x_anchor, batch, log_values={}, loss_prefix=''):

        points_action = batch['points_action'][:, :, :3]
        points_anchor = batch['points_anchor'][:, :, :3]
        points_trans_action = batch['points_action_trans'][:, :, :3]
        points_trans_anchor = batch['points_anchor_trans'][:, :, :3]

        T0 = Transform3d(matrix=batch['T0'])
        T1 = Transform3d(matrix=batch['T1'])

        R0_max, R0_min, R0_mean = get_degree_angle(T0)
        R1_max, R1_min, R1_mean = get_degree_angle(T1)
        t0_max, t0_min, t0_mean = get_translation(T0)
        t1_max, t1_min, t1_mean = get_translation(T1)

        pred_flow_action, pred_w_action = self.extract_flow_and_weight(
            x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
            x_anchor)

        pred_T_action = dualflow2pose(xyz_src=points_trans_action, xyz_tgt=points_trans_anchor,
                                      flow_src=pred_flow_action, flow_tgt=pred_flow_anchor,
                                      weights_src=pred_w_action, weights_tgt=pred_w_anchor,
                                      return_transform3d=True, normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)

        induced_flow_action = (pred_T_action.transform_points(
            points_trans_action) - points_trans_action).detach()
        pred_points_action = pred_T_action.transform_points(
            points_trans_action)

        # pred_T_action=T1T0^-1
        gt_T_action = T0.inverse().compose(T1)
        points_action_target = T1.transform_points(points_action)

        error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))

        error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))

        # Loss associated with ground truth transform
        point_loss_action = mse_criterion(
            pred_points_action,
            points_action_target,
        )

        point_loss = self.action_weight * point_loss_action

        dense_loss = dense_flow_loss(points=points_trans_action,
                                     flow_pred=pred_flow_action,
                                     trans_gt=gt_T_action)

        # Loss associated flow vectors matching a consistent rigid transform
        smoothness_loss_action = mse_criterion(
            pred_flow_action,
            induced_flow_action,
        )

        smoothness_loss = self.action_weight * smoothness_loss_action

        loss = point_loss + self.smoothness_weight * \
            smoothness_loss + self.consistency_weight * dense_loss

        log_values[loss_prefix+'point_loss'] = point_loss
        log_values[loss_prefix +
                   'rotation_loss'] = self.rotation_weight * error_R_mean

        log_values[loss_prefix +
                   'smoothness_loss'] = self.smoothness_weight * smoothness_loss
        log_values[loss_prefix +
                   'dense_loss'] = self.consistency_weight * dense_loss

        log_values[loss_prefix+'R0_mean'] = R0_mean
        log_values[loss_prefix+'R0_max'] = R0_max
        log_values[loss_prefix+'R0_min'] = R0_min
        log_values[loss_prefix+'R1_mean'] = R1_mean
        log_values[loss_prefix+'R1_max'] = R1_max
        log_values[loss_prefix+'R1_min'] = R1_min

        log_values[loss_prefix+'t0_mean'] = t0_mean
        log_values[loss_prefix+'t0_max'] = t0_max
        log_values[loss_prefix+'t0_min'] = t0_min
        log_values[loss_prefix+'t1_mean'] = t1_mean
        log_values[loss_prefix+'t1_max'] = t1_max
        log_values[loss_prefix+'t1_min'] = t1_min

        log_values[loss_prefix+'error_R_mean'] = error_R_mean
        log_values[loss_prefix+'error_t_mean'] = error_t_mean

        return loss, log_values

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if(x.shape[2] > 3):
            if self.sigmoid_on:
                pred_w = torch.sigmoid(x[:, :, 3])
            else:
                pred_w = x[:, :, 3]
        else:
            pred_w = None
        return pred_flow, pred_w

    def module_step(self, batch, batch_idx):
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']

        T0 = Transform3d(matrix=batch['T0'])
        T1 = Transform3d(matrix=batch['T1'])
        if self.return_flow_component:
            model_output = self.model(points_trans_action, points_trans_anchor)
            x_action = model_output['flow_action']
            x_anchor = model_output['flow_anchor']
            residual_flow_action = model_output['residual_flow_action']
            residual_flow_anchor = model_output['residual_flow_anchor']
            corr_flow_action = model_output['corr_flow_action']
            corr_flow_anchor = model_output['corr_flow_anchor']
        else:
            x_action, x_anchor = self.model(
                points_trans_action, points_trans_anchor)

        log_values = {}
        loss, log_values = self.compute_loss(
            x_action, x_anchor, batch, log_values=log_values, loss_prefix='')
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        # classes = batch['classes']
        # points = batch['points']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']
        # points_trans = batch['points_trans']
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']

        T0 = Transform3d(matrix=batch['T0'])
        T1 = Transform3d(matrix=batch['T1'])

        if self.return_flow_component:
            model_output = self.model(points_trans_action, points_trans_anchor)
            x_action = model_output['flow_action']
            x_anchor = model_output['flow_anchor']
            residual_flow_action = model_output['residual_flow_action']
            residual_flow_anchor = model_output['residual_flow_anchor']
            corr_flow_action = model_output['corr_flow_action']
            corr_flow_anchor = model_output['corr_flow_anchor']
        else:
            x_action, x_anchor = self.model(
                points_trans_action, points_trans_anchor)

        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]

        pred_flow_action = x_action[:, :, :3]
        if(x_action .shape[2] > 3):
            if self.sigmoid_on:
                pred_w_action = torch.sigmoid(x_action[:, :, 3])
            else:
                pred_w_action = x_action[:, :, 3]
        else:
            pred_w_action = None

        pred_flow_anchor = x_anchor[:, :, :3]
        if(x_anchor .shape[2] > 3):
            if self.sigmoid_on:
                pred_w_anchor = torch.sigmoid(x_anchor[:, :, 3])
            else:
                pred_w_anchor = x_anchor[:, :, 3]
        else:
            pred_w_anchor = None

        pred_T_action = dualflow2pose(xyz_src=points_trans_action, xyz_tgt=points_trans_anchor,
                                      flow_src=pred_flow_action, flow_tgt=pred_flow_anchor,
                                      weights_src=pred_w_action, weights_tgt=pred_w_anchor,
                                      return_transform3d=True, normalization_scehme=self.weight_normalize,
                                      temperature=self.softmax_temperature)

        pred_points_action = pred_T_action.transform_points(
            points_trans_action)
        points_action_target = T1.transform_points(points_action)

        res_images = {}

        demo_points = get_color(
            tensor_list=[points_action[0], points_anchor[0]], color_list=['blue', 'red'])
        res_images['demo_points'] = wandb.Object3D(
            demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]], color_list=['blue', 'red'])
        res_images['action_transformed_action'] = wandb.Object3D(
            action_transformed_action)

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images['anchor_transformed_anchor'] = wandb.Object3D(
            anchor_transformed_anchor)

        transformed_input_points = get_color(tensor_list=[
            points_trans_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images['transformed_input_points'] = wandb.Object3D(
            transformed_input_points)

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images['demo_points_apply_action_transform'] = wandb.Object3D(
            demo_points_apply_action_transform)

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'red'])
        res_images['apply_action_transform_demo_comparable'] = wandb.Object3D(
            apply_action_transform_demo_comparable)

        predicted_vs_gt_transform_applied = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], points_action[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'green', 'red', ])
        res_images['predicted_vs_gt_transform_applied'] = wandb.Object3D(
            predicted_vs_gt_transform_applied)

        apply_predicted_transform = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'orange', 'red', ])
        res_images['apply_predicted_transform'] = wandb.Object3D(
            apply_predicted_transform)

        loss_points_action = get_color(
            tensor_list=[points_action_target[0], pred_points_action[0]], color_list=['green', 'red'])
        res_images['loss_points_action'] = wandb.Object3D(
            loss_points_action)

        return res_images
