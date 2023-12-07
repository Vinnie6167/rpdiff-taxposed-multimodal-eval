from equivariant_pose_graph.models.transformer_flow import ResidualFlow_DiffEmbTransformer
from equivariant_pose_graph.models.multimodal_transformer_flow import Multimodal_ResidualFlow_DiffEmbTransformer, Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX
from equivariant_pose_graph.training.flow_equivariance_training_module_nocentering_eval_init import EquivarianceTestingModule
from equivariant_pose_graph.training.flow_equivariance_training_module_nocentering_multimodal import EquivarianceTrainingModule, EquivarianceTrainingModule_WithPZCondX
import torch

def load_model(place_checkpoint_file, has_pzX, conditioning="pos_delta_l2norm", return_flow_component=None, cfg=None):
    if cfg is not None:
        print("Loading model with architecture specified in configs")
        TP_input_dims = Multimodal_ResidualFlow_DiffEmbTransformer.TP_INPUT_DIMS[cfg.conditioning]

        inner_network = ResidualFlow_DiffEmbTransformer(
            emb_dims=cfg.emb_dims,
            input_dims=TP_input_dims,
            emb_nn=cfg.emb_nn,
            return_flow_component=cfg.return_flow_component if return_flow_component is None else return_flow_component,
            center_feature=cfg.center_feature,
            inital_sampling_ratio=cfg.inital_sampling_ratio,
            pred_weight=cfg.pred_weight,
            freeze_embnn=cfg.freeze_embnn,
            conditioning_size=cfg.latent_z_linear_size if cfg.conditioning in ["latent_z_linear_internalcond"] else 0,
            )

        place_nocond_network = Multimodal_ResidualFlow_DiffEmbTransformer(
            residualflow_diffembtransformer=inner_network,
            gumbel_temp=cfg.gumbel_temp,
            freeze_residual_flow=cfg.freeze_residual_flow,
            center_feature=cfg.center_feature,
            freeze_z_embnn=cfg.freeze_z_embnn,
            division_smooth_factor=cfg.division_smooth_factor,
            add_smooth_factor=cfg.add_smooth_factor,
            conditioning=cfg.conditioning,
            latent_z_linear_size=cfg.latent_z_linear_size,
            taxpose_centering=cfg.taxpose_centering
        )

        place_nocond_model = EquivarianceTrainingModule(
            place_nocond_network,
            lr=cfg.lr,
            # image_log_period=cfg.image_logging_period,
            point_loss_type=cfg.point_loss_type,
            rotation_weight=cfg.rotation_weight,
            weight_normalize=cfg.weight_normalize,
            consistency_weight=cfg.consistency_weight,
            smoothness_weight=cfg.smoothness_weight,
            action_weight=cfg.action_weight,
            #latent_weight=cfg.latent_weight,
            vae_reg_loss_weight=cfg.vae_reg_loss_weight,
            sigmoid_on=cfg.sigmoid_on,
            softmax_temperature=cfg.softmax_temperature,
            min_err_across_racks_debug=cfg.min_err_across_racks_debug,
            error_mode_2rack=cfg.error_mode_2rack)
        
        if has_pzX:
            network_cond_x = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
                                residualflow_embnn=place_nocond_network,
                                encoder_type=cfg.pzcondx_encoder_type,
            )
            place_model = EquivarianceTrainingModule_WithPZCondX(
                network_cond_x,
                place_nocond_model,
                goal_emb_cond_x_loss_weight=cfg.goal_emb_cond_x_loss_weight,
            )
        else:
            place_model = place_nocond_model
    else:
        print("WARNING: No configs for the model are specified. Using default configs to load the model")
        inner_network = ResidualFlow_DiffEmbTransformer(
                    emb_nn='dgcnn', return_flow_component=return_flow_component, center_feature=True,
                    inital_sampling_ratio=1, input_dims=4)
        place_nocond_network = Multimodal_ResidualFlow_DiffEmbTransformer(
                            inner_network, gumbel_temp=1, center_feature=True, conditioning=conditioning)
        place_nocond_model = EquivarianceTrainingModule(
            place_nocond_network,
            lr=1e-4,
            image_log_period=100,
            weight_normalize='softmax', #'l1',
            softmax_temperature=1
        )
        
        if has_pzX:
            place_network = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
                            place_nocond_network, encoder_type="2_dgcnn", sample_z=False)
            place_model = EquivarianceTrainingModule_WithPZCondX(
                place_network,
                place_nocond_model,
            )
        else:
            place_model = place_nocond_model

    place_model.cuda()

    if place_checkpoint_file is not None:
        place_model.load_state_dict(torch.load(place_checkpoint_file)['state_dict'])
    else:
        print("WARNING: NO CHECKPOINT FILE SPECIFIED. THIS IS A DEBUG RUN WITH RANDOM WEIGHTS")
    return place_model

def load_merged_model(place_checkpoint_file_pzY, place_checkpoint_file_pzX, conditioning="pos_delta_l2norm", return_flow_component=None, cfg=None):
    if place_checkpoint_file_pzY is not None:
        pzY_model = load_model(place_checkpoint_file_pzY, has_pzX=False, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
    
        if place_checkpoint_file_pzX is not None:
            pzX_model = load_model(place_checkpoint_file_pzX, has_pzX=True, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
            pzX_model.model.tax_pose = pzY_model.model.tax_pose
            return pzX_model
        else:
            return pzY_model
    else:
        if place_checkpoint_file_pzX is not None:
            pzX_model = load_model(place_checkpoint_file_pzX, has_pzX=True, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
            return pzX_model
        else:
            raise ValueError("No checkpoint file specified for either pzY or pzX. Cannot load a model")