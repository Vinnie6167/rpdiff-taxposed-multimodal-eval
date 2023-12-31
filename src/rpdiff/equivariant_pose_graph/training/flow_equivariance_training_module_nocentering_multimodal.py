import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch3d.transforms import Transform3d, Translate, matrix_to_axis_angle
from torchvision.transforms import ToTensor
from pytorch3d.loss import chamfer_distance
from equivariant_pose_graph.training.point_cloud_training_module import PointCloudTrainingModule
from equivariant_pose_graph.utils.se3 import dualflow2pose, flow2pose, get_translation, get_degree_angle, dense_flow_loss, pure_translation_se3
from equivariant_pose_graph.utils.display_headless import scatter3d, quiver3d
from equivariant_pose_graph.utils.color_utils import get_color, color_gradient
from equivariant_pose_graph.utils.error_metrics import get_2rack_errors

import torch.nn.functional as F

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
                 latent_weight=0.1,
                 vae_reg_loss_weight=0.01,
                 rotation_weight=0,
                 chamfer_weight=10000,
                 point_loss_type=0,
                 return_flow_component=False,
                 weight_normalize='l1',
                 sigmoid_on=False,
                 softmax_temperature=None,
                 min_err_across_racks_debug=False,
                 error_mode_2rack="batch_min_rack"):
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
        self.latent_weight = latent_weight
        # This is only used when the internal model uses self.model.conditioning == "latent_z" or latent_z_1pred or latent_z_1pred_10d
        self.vae_reg_loss_weight = vae_reg_loss_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize
        # 0 for mse loss, 1 for chamfer distance, 2 for mse loss + chamfer distance
        self.point_loss_type = point_loss_type
        self.return_flow_component = return_flow_component
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        self.min_err_across_racks_debug = min_err_across_racks_debug
        self.error_mode_2rack = error_mode_2rack
        if self.weight_normalize == 'l1':
            assert (self.sigmoid_on), "l1 weight normalization need sigmoid on"

    def action_centered(self, points_action, points_anchor):
        """
        @param points_action, (1,num_points,3)
        @param points_anchor, (1,num_points,3)
        """
        points_action_mean = points_action.clone().mean(axis=1)
        points_action_mean_centered = points_action-points_action_mean
        points_anchor_mean_centered = points_anchor - points_action_mean

        return points_action_mean_centered, points_anchor_mean_centered, points_action_mean

    def get_transform(self, points_trans_action, points_trans_anchor, points_onetrans_action=None, points_onetrans_anchor=None, mode="forward"):
        if(self.model.return_flow_component):
            res = self.model(
               points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode=mode)
            x_action = res['flow_action']
            x_anchor = res['flow_anchor']
        else:
            if self.model.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d",]:
                x_action, x_anchor, goal_emb = self.model(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode=mode)
                heads = None
            else:
                x_action, x_anchor, goal_emb, heads = self.model_with_cond_x(
                    mode, points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode=mode)

        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]
        ans_dict = self.predict(x_action=x_action, x_anchor=x_anchor,
                                points_trans_action=points_trans_action, points_trans_anchor=points_trans_anchor)
        if(self.model.return_flow_component):
            ans_dict['flow_components'] = res
        return ans_dict

    def predict(self, x_action, x_anchor, points_trans_action, points_trans_anchor):
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(
            x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
            x_anchor)

        pred_T_action = dualflow2pose(xyz_src=points_trans_action, xyz_tgt=points_trans_anchor,
                                      flow_src=pred_flow_action, flow_tgt=pred_flow_anchor,
                                      weights_src=pred_w_action, weights_tgt=pred_w_anchor,
                                      return_transform3d=True, normalization_scehme=self.weight_normalize, temperature=self.softmax_temperature)
        pred_points_action = pred_T_action.transform_points(
            points_trans_action)

        return {"pred_T_action": pred_T_action,
                "pred_points_action": pred_points_action}

    def compute_loss(self, x_action, x_anchor, goal_emb, batch, log_values={}, loss_prefix='', heads=None):
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

        if not self.min_err_across_racks_debug:
            # don't print rotation/translation error metrics to the logs
            pass
            # error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
            #     T1).compose(pred_T_action.inverse()))

            # error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
            #     T1).compose(pred_T_action.inverse()))
        else:
            error_R_mean, error_t_mean = get_2rack_errors(pred_T_action, T0, T1, mode=self.error_mode_2rack)
            log_values[loss_prefix+'error_R_mean'] = error_R_mean
            log_values[loss_prefix+'error_t_mean'] = error_t_mean
            log_values[loss_prefix +
                   'rotation_loss'] = self.rotation_weight * error_R_mean

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

        #uniform_latent = torch.full_like(goal_emb, 1/goal_emb.shape[-1])
        #goal_emb_norm = (goal_emb - goal_emb.max(axis=-1, keepdim=True)[0]).exp()
        #goal_emb_norm = goal_emb_norm / goal_emb_norm.sum(axis=-1, keepdim=True)
        #latent_loss = self.latent_weight * torch.nn.functional.kl_div(goal_emb_norm, uniform_latent, log_target=True, reduction='batchmean')

        #uniform_latent = torch.randn_like(goal_emb)
        #latent_loss = self.latent_weight * torch.nn.functional.kl_div(goal_emb, uniform_latent, log_target=True, reduction='batchmean')

        loss = point_loss + self.smoothness_weight * \
            smoothness_loss + self.consistency_weight * dense_loss #+ latent_loss

        log_values[loss_prefix+'point_loss'] = point_loss

        log_values[loss_prefix +
                   'smoothness_loss'] = self.smoothness_weight * smoothness_loss
        log_values[loss_prefix +
                   'dense_loss'] = self.consistency_weight * dense_loss
        #log_values[loss_prefix +
        #           'latent_loss'] = self.latent_weight * latent_loss

        if self.model.conditioning in ["uniform_prior_pos_delta_l2norm"]:
            N = x_action.shape[1]
            uniform = (torch.ones((goal_emb.shape[0], goal_emb.shape[1], N)) / goal_emb.shape[-1]).cuda().detach()
            action_kl = F.kl_div(F.log_softmax(uniform, dim=-1),
                                            F.log_softmax(goal_emb[:, :, :N], dim=-1), log_target=True,
                                            reduction='batchmean')
            anchor_kl = F.kl_div(F.log_softmax(uniform, dim=-1),
                                            F.log_softmax(goal_emb[:, :, N:], dim=-1), log_target=True,
                                            reduction='batchmean')
            vae_reg_loss = action_kl + anchor_kl
            loss += self.vae_reg_loss_weight * vae_reg_loss
            log_values[loss_prefix+'vae_reg_loss'] = self.vae_reg_loss_weight * vae_reg_loss

        if heads is not None:
            def vae_regularization_loss(mu, log_var):
                # From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/cvae.py#LL144C9-L144C105
                return torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim = -1).sum(dim = -1), dim = 0)
            
            if self.model.conditioning in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                vae_reg_loss = vae_regularization_loss(heads['goal_emb_mu'], heads['goal_emb_logvar'])
                vae_reg_loss = torch.nan_to_num(vae_reg_loss)
                
                loss += self.vae_reg_loss_weight * vae_reg_loss
                log_values[loss_prefix+'vae_reg_loss'] = self.vae_reg_loss_weight * vae_reg_loss
            else:
                raise ValueError("ERROR: Why is there a non-None heads variable passed in when the model isn't even a latent_z model?")

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
        points_onetrans_action = batch['points_action_onetrans']
        points_onetrans_anchor = batch['points_anchor_onetrans']
        # points_action = batch['points_action']
        # points_anchor = batch['points_anchor']

        if self.return_flow_component:
            # TODO only pass in points_anchor and points_action if the model is training
            model_output = self.model(points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
            x_action = model_output['flow_action']
            x_anchor = model_output['flow_anchor']
            goal_emb = model_output['goal_emb']
            # residual_flow_action = model_output['residual_flow_action']
            # residual_flow_anchor = model_output['residual_flow_anchor']
            # corr_flow_action = model_output['corr_flow_action']
            # corr_flow_anchor = model_output['corr_flow_anchor']
        else:
            if self.model.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                x_action, x_anchor, goal_emb = self.model(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
                heads = None
            else:
                x_action, x_anchor, goal_emb, heads = self.model(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)

        log_values = {}
        loss, log_values = self.compute_loss(
            x_action, x_anchor, goal_emb, batch, log_values=log_values, loss_prefix='', heads=heads)
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            def get_inference_error(log_values, batch, loss_prefix):
                T0 = Transform3d(matrix=batch['T0'])
                T1 = Transform3d(matrix=batch['T1'])

                if self.return_flow_component:
                    model_output = self.model(points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode="inference")
                    x_action = model_output['flow_action']
                    x_anchor = model_output['flow_anchor']
                else:
                    if self.model.conditioning not in ["uniform_prior_pos_delta_l2norm", "latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                        # not inference mode here because these models don't have a separate inference mode
                        x_action, x_anchor, goal_emb = self.model(
                            points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode="forward")
                    else:
                        x_action, x_anchor, goal_emb = self.model(
                            points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor, mode="inference")
                        
                pred_flow_action, pred_w_action = self.extract_flow_and_weight(
                    x_action)
                pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
                    x_anchor)
                
                del x_action, x_anchor, goal_emb

                pred_T_action = dualflow2pose(xyz_src=points_trans_action, xyz_tgt=points_trans_anchor,
                                            flow_src=pred_flow_action, flow_tgt=pred_flow_anchor,
                                            weights_src=pred_w_action, weights_tgt=pred_w_anchor,
                                            return_transform3d=True, normalization_scehme=self.weight_normalize,
                                            temperature=self.softmax_temperature)

                if not self.min_err_across_racks_debug:
                    # don't print rotation/translation error metrics to the logs
                    pass
                    # error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
                    #     T1).compose(pred_T_action.inverse()))

                    # error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
                    #     T1).compose(pred_T_action.inverse()))
                else:
                    error_R_mean, error_t_mean = get_2rack_errors(pred_T_action, T0, T1, mode=self.error_mode_2rack)
                    log_values[loss_prefix+'sample_error_R_mean'] = error_R_mean
                    log_values[loss_prefix+'sample_error_t_mean'] = error_t_mean
            get_inference_error(log_values, batch, loss_prefix='')
        torch.cuda.empty_cache()

        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        # classes = batch['classes']
        # points = batch['points']

        # points_trans = batch['points_trans']
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']
        points_onetrans_action = batch['points_action_onetrans']
        points_onetrans_anchor = batch['points_anchor_onetrans']

        T0 = Transform3d(matrix=batch['T0'])
        T1 = Transform3d(matrix=batch['T1'])

        if self.return_flow_component:
            model_output = self.model(points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
            x_action = model_output['flow_action']
            x_anchor = model_output['flow_anchor']
            goal_emb = model_output['goal_emb']
            residual_flow_action = model_output['residual_flow_action']
            residual_flow_anchor = model_output['residual_flow_anchor']
            corr_flow_action = model_output['corr_flow_action']
            corr_flow_anchor = model_output['corr_flow_anchor']
        else:
            if self.model.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                x_action, x_anchor, goal_emb = self.model(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
                heads = None
            else:
                x_action, x_anchor, goal_emb, heads = self.model(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)

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
            tensor_list=[points_onetrans_action[0], points_onetrans_anchor[0]], color_list=['blue', 'red'])
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

        # transformed_input_points = get_color(tensor_list=[
        #     points_trans_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        # res_images['transformed_input_points'] = wandb.Object3D(
        #     transformed_input_points)

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

        # loss_points_action = get_color(
        #     tensor_list=[points_action_target[0], pred_points_action[0]], color_list=['green', 'red'])
        # res_images['loss_points_action'] = wandb.Object3D(
        #     loss_points_action)

        if self.model.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            goal_emb_norm_action = F.softmax(goal_emb[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
            goal_emb_norm_anchor = F.softmax(goal_emb[0, :, points_action.shape[1]:], dim=-1).detach().cpu()
            colors_action = color_gradient(goal_emb_norm_action[0])
            colors_anchor = color_gradient(goal_emb_norm_anchor[0])
            goal_emb_on_objects = np.concatenate([
                torch.cat([points_action[0].detach(), points_anchor[0].detach()], dim=0).cpu().numpy(),
                np.concatenate([colors_action, colors_anchor], axis=0)],
                axis=-1)
            res_images['goal_emb'] = wandb.Object3D(
                goal_emb_on_objects)

        return res_images



class EquivarianceTrainingModule_WithPZCondX(PointCloudTrainingModule):

    def __init__(self,
                 model_with_cond_x,
                 training_module_no_cond_x,
                 goal_emb_cond_x_loss_weight=1,
                 pzy_pzx_loss_type="reverse_kl"):
        # TODO add this in
        assert pzy_pzx_loss_type in ["reverse_kl", "forward_kl", "mse"]

        super().__init__(model=model_with_cond_x, lr=training_module_no_cond_x.lr, 
                         image_log_period=training_module_no_cond_x.image_log_period)

        self.model_with_cond_x = model_with_cond_x
        self.model = self.model_with_cond_x.residflow_embnn
        self.training_module_no_cond_x = training_module_no_cond_x
        self.goal_emb_cond_x_loss_weight = goal_emb_cond_x_loss_weight
        

    def action_centered(self, points_action, points_anchor):
        """
        @param points_action, (1,num_points,3)
        @param points_anchor, (1,num_points,3)
        """
        points_action_mean = points_action.clone().mean(axis=1)
        points_action_mean_centered = points_action-points_action_mean
        points_anchor_mean_centered = points_anchor - points_action_mean

        return points_action_mean_centered, points_anchor_mean_centered, points_action_mean

    def get_transform(self, points_trans_action, points_trans_anchor, points_action=None, points_anchor=None, mode="forward"):
        # mode is unused

        if(self.model.return_flow_component):
            res = self.model_with_cond_x(
                points_trans_action, points_trans_anchor, points_action, points_anchor)
            x_action = res['flow_action']
            x_anchor = res['flow_anchor']
        else:
            if self.model_with_cond_x.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                x_action, x_anchor, goal_emb, goal_emb_cond_x = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_action, points_anchor)
                heads = None
            else:
                x_action, x_anchor, goal_emb, goal_emb_cond_x, heads = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_action, points_anchor)

        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]
        ans_dict = self.predict(x_action=x_action, x_anchor=x_anchor,
                                points_trans_action=points_trans_action, points_trans_anchor=points_trans_anchor)
        if(self.model.return_flow_component):
            ans_dict['flow_components'] = res
        return ans_dict

    def predict(self, x_action, x_anchor, points_trans_action, points_trans_anchor):
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(
            x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(
            x_anchor)

        pred_T_action = dualflow2pose(xyz_src=points_trans_action, xyz_tgt=points_trans_anchor,
                                      flow_src=pred_flow_action, flow_tgt=pred_flow_anchor,
                                      weights_src=pred_w_action, weights_tgt=pred_w_anchor,
                                      return_transform3d=True, normalization_scehme=self.training_module_no_cond_x.weight_normalize, temperature=self.training_module_no_cond_x.softmax_temperature)
        pred_points_action = pred_T_action.transform_points(
            points_trans_action)

        return {"pred_T_action": pred_T_action,
                "pred_points_action": pred_points_action}

    def compute_loss(self, x_action, x_anchor, goal_emb, goal_emb_cond_x, batch, log_values={}, loss_prefix=''):
        loss, log_values = self.training_module_no_cond_x.compute_loss(x_action, x_anchor, goal_emb, batch, log_values, loss_prefix)
        # aka "if it is training time and not val time"
        if goal_emb is not None:
            B, K, D = goal_emb.shape

            if self.model_with_cond_x.conditioning != "latent_z_linear_internalcond":
                N = x_action.shape[1]
                action_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, :N], dim=-1),
                                                F.log_softmax(goal_emb[:, :, :N], dim=-1), log_target=True,
                                                reduction='batchmean')
                anchor_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, N:], dim=-1),
                                                F.log_softmax(goal_emb[:, :, N:], dim=-1), log_target=True,
                                                reduction='batchmean')
            else:
                goal_emb_cond_x = goal_emb_cond_x[0] # just take the mean
                action_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, 0], dim=-1),
                                            F.log_softmax(goal_emb[:, :, 0], dim=-1), log_target=True,
                                            reduction='batchmean')
                anchor_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, 1], dim=-1),
                                            F.log_softmax(goal_emb[:, :, 0], dim=-1), log_target=True,
                                            reduction='batchmean')
                
            goal_emb_loss = action_kl + anchor_kl

            
            if self.model_with_cond_x.freeze_residual_flow and self.model_with_cond_x.freeze_z_embnn and self.model_with_cond_x.freeze_embnn:
                # Overwrite the loss because the other losses are not used
                loss = self.goal_emb_cond_x_loss_weight * goal_emb_loss
            else:
                # DON'T overwrite the loss
                loss += self.goal_emb_cond_x_loss_weight * goal_emb_loss

            log_values[loss_prefix+'goal_emb_cond_x_loss'] = self.goal_emb_cond_x_loss_weight * goal_emb_loss
            log_values[loss_prefix+'action_kl'] = action_kl
            log_values[loss_prefix + 'anchor_kl'] = anchor_kl

        return loss, log_values

    def extract_flow_and_weight(self, *args, **kwargs):
        return self.training_module_no_cond_x.extract_flow_and_weight(*args, **kwargs)

    def module_step(self, batch, batch_idx):
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']

        points_onetrans_action = batch['points_action_onetrans']
        points_onetrans_anchor = batch['points_anchor_onetrans']

        if self.model.tax_pose.return_flow_component:
            model_output = self.model_with_cond_x(points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
            x_action = model_output['flow_action']
            x_anchor = model_output['flow_anchor']
            goal_emb = model_output['goal_emb']
            goal_emb_cond_x = model_output['goal_emb_cond_x']
        else:
            if self.model_with_cond_x.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                x_action, x_anchor, goal_emb, goal_emb_cond_x = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
                heads = None
            else:
                x_action, x_anchor, goal_emb, goal_emb_cond_x, heads = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)

        log_values = {}
        loss, log_values = self.compute_loss(
            x_action, x_anchor, goal_emb, goal_emb_cond_x, batch, log_values=log_values, loss_prefix='')
        return loss, log_values
    
    def visualize_results(self, batch, batch_idx):
        res_images = self.training_module_no_cond_x.visualize_results(batch, batch_idx)

        # This goal_emb_cond_x visualization only applies to methods that have a per-point latent space
        if self.model.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            points_trans_action = batch['points_action_trans']
            points_trans_anchor = batch['points_anchor_trans']
            points_action = batch['points_action']
            points_anchor = batch['points_anchor']
            points_onetrans_action = batch['points_action_onetrans']
            points_onetrans_anchor = batch['points_anchor_onetrans']

            if self.training_module_no_cond_x.return_flow_component:
                model_output = self.model_with_cond_x(points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
                goal_emb_cond_x = model_output['goal_emb_cond_x']
            else:
                if self.model_with_cond_x.conditioning not in ["latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear", "latent_z_linear_internalcond"]:
                    x_action, x_anchor, goal_emb, goal_emb_cond_x = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)
                    heads = None
                else:
                    x_action, x_anchor, goal_emb, goal_emb_cond_x, heads = self.model_with_cond_x(
                    points_trans_action, points_trans_anchor, points_onetrans_action, points_onetrans_anchor)

            points_action = points_action[:, :, :3]
            points_anchor = points_anchor[:, :, :3]

            goal_emb_norm_action = F.softmax(goal_emb_cond_x[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
            goal_emb_norm_anchor = F.softmax(goal_emb_cond_x[0, :, points_action.shape[1]:], dim=-1).detach().cpu()

            # TODO CHANGE THIS. temporary only mug
            only_mug = False
            if only_mug:
                colors_action = color_gradient(F.softmax(goal_emb_norm_action[0], dim=-1))
                points = points_action[0].detach().cpu().numpy()
                goal_emb_on_objects = np.concatenate([
                    points,
                    colors_action],
                    axis=-1)
            else:
                colors_action = color_gradient(F.softmax(goal_emb_norm_action[0], dim=-1))
                colors_anchor = color_gradient(F.softmax(goal_emb_norm_anchor[0], dim=-1))
                points = torch.cat([points_action[0].detach(), points_anchor[0].detach()], dim=0).cpu().numpy()
                goal_emb_on_objects = np.concatenate([
                    points,
                    np.concatenate([colors_action, colors_anchor], axis=0)],
                    axis=-1)
                
            min_range = points[:,:3].min(dim=0).values.cpu().numpy()
            max_range = points[:,:3].max(dim=0).values.cpu().numpy()
            range_size = max(max_range - min_range)
            marker_scale = 0.01  # Adjust this value based on your preference
            res_images['goal_emb_cond_x'] = wandb.Object3D(
                goal_emb_on_objects, markerSize=1000) #marker_scale * range_size)

        return res_images
