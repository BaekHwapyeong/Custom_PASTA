# CUDA_VISIBLE_DEVICES=1 python your_script.py

import argparse
import os

from src.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from src.diffusion.respace import *
from src.trainer.trainer import Trainer
from src.model.unet import UNetModel
from src.utils.utils import *

OBJECTIVE = {'PREVIOUS_X': ModelMeanType.PREVIOUS_X, 'START_X': ModelMeanType.START_X, 'EPSILON': ModelMeanType.EPSILON, 'VELOCITY': ModelMeanType.VELOCITY}
MODEL_VAR_TYPE = {'LEARNED': ModelVarType.LEARNED, 'FIXED_SMALL': ModelVarType.FIXED_SMALL, 'FIXED_LARGE': ModelVarType.FIXED_LARGE, 'LEARNED_RANGE': ModelVarType.LEARNED_RANGE}
LOSS_TYPE = {'l1': LossType.MAE, 'l2': LossType.MSE}

config_file = 'pasta_mri2pet.yaml'

config_root = '/mnt/peace_2/BHP/PASTA/src/config'
# config_root = './PASTA/src/config'
config_path = os.path.join(config_root, config_file)

def main():
    args = load_config_from_yaml(config_path)
    args_dict = args.__dict__
    
    if not os.path.exists(args_dict['results_folder']):
        os.makedirs(args_dict['results_folder'])

    list_of_dict = [ f'{key} : {args_dict[key]}' for key in args_dict ]

    if not args_dict['eval_mode']:
        with open(os.path.join(args_dict['results_folder'], '_hyperparameters.yaml'), 'w') as data:
            [data.write(f'{st}\n') for st in list_of_dict]
  
    # Add support for fMRI and FC conditioning in UNet if needed
    # Check if fMRI/FC conditioning is enabled in config
    use_fmri_cond = getattr(args, 'use_fmri_cond', False)
    use_fc_cond = getattr(args, 'use_fc_cond', False)
    
    # Calculate additional input channels for conditioning
    additional_cond_channels = 0
    if use_fmri_cond:
        additional_cond_channels += 1  # fMRI will be processed separately
    if use_fc_cond:
        additional_cond_channels += 1  # FC will be processed separately
    
    model = UNetModel(
        image_size = args.image_size,
        in_channels = args.model_in_channels,
        model_channels = args.unet_dim,
        out_channels = args.out_channels_model,
        num_res_blocks = args.num_res_blocks,
        attention_resolutions = args.attention_resolutions, # [16],
        num_heads = args.num_heads,
        channel_mult = args.unet_dim_mults,
        resblock_updown = args.resblock_updown,
        dims=args.dims,
        dropout = args.dropout,
        use_fp16 = False, 
        use_scale_shift_norm = True,
        use_condition = True,
        use_time_condition = args.use_time_condition,
        cond_emb_channels = args.cond_emb_channels,
        tab_cond_dim = args.tab_cond_dim,
        use_tabular_cond = args.use_tabular_cond_model,
        with_attention = args.with_attention,
        cond_apply_method = args.cond_apply_method,
        # Add new conditioning parameters if available in config
        use_fmri_cond = use_fmri_cond,
        use_fc_cond = use_fc_cond,
    )

    encoder = UNetModel(
        image_size = args.image_size,
        in_channels = args.encoder_in_channels,
        model_channels = args.unet_dim,
        out_channels = args.out_channels_encoder,
        num_res_blocks = args.num_res_blocks,
        attention_resolutions = args.attention_resolutions, # [16],
        num_heads = args.num_heads,
        channel_mult = args.unet_dim_mults,
        resblock_updown = args.resblock_updown,
        dims=args.dims,
        dropout = args.dropout,
        use_fp16 = False, 
        use_scale_shift_norm = True,
        use_condition = True,
        use_time_condition = args.use_time_condition,
        tab_cond_dim = args.tab_cond_dim,
        use_tabular_cond = args.use_tabular_cond_encoder,
        with_attention = args.with_attention,
        cond_apply_method = args.cond_apply_method,
        # Add new conditioning parameters if available in config
        use_fmri_cond = use_fmri_cond,
        use_fc_cond = use_fc_cond,
    )


    # spaced diffusion for ddim
    diffusion = SpacedDiffusion(
        use_timesteps = space_timesteps(args.timesteps, args.timestep_respacing),
        model = model,
        encoder = encoder,
        beta_schedule=args.beta_schedule,
        timesteps=args.timesteps,
        model_mean_type = OBJECTIVE[args.objective],
        model_var_type = MODEL_VAR_TYPE[args.model_var_type],
        loss_type = LOSS_TYPE[args.loss_type],
        gen_type = args.gen_type,
        use_fp16 = False, 
        condition = args.condition,
        reconstructed_loss = args.reconstructed_loss,
        recon_weight = args.recon_weight,
        rescale_intensity = args.rescale_intensity,
    )

    # Get fMRI and FC usage flags from config, default to False if not specified
    use_fmri = getattr(args, 'use_fmri', False)
    use_fc = getattr(args, 'use_fc', False)
    
    trainer = Trainer(
        diffusion,
        folder = args.data_dir,
        input_slice_channel = args.input_slice_channel,
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps,         # total training steps
        save_and_sample_every = args.save_and_sample_every,    # every n steps, save checkpoint & sample generative images
        num_samples = args.num_samples,
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = args.ema_decay,                # exponential moving average decay
        amp = args.amp,                       # turn on mixed precision
        fp16 = args.fp16,
        calculate_fid = args.calculate_fid,              # whether to calculate fid during training
        dataset = args.dataset,                  # dataset name
        image_direction = args.image_direction,
        num_slices = args.num_slices,
        tabular_cond = args.tabular_cond,
        results_folder = args.results_folder,
        resume = args.resume,
        pretrain = args.pretrain,
        test_batch_size = args.test_batch_size,
        eval_mode = args.eval_mode,
        eval_dataset = args.eval_dataset,
        eval_resolution = args.eval_resolution,
        model_cycling = args.model_cycling,
        ROI_mask = args.ROI_mask,
        dx_labels = args.dx_labels,
        # Add new parameters for fMRI and FC support
        use_fmri = use_fmri,
        use_fc = use_fc,
    )

    if trainer.eval_mode:
        if args.synthesis:
            synth_folder = os.path.join(trainer.results_folder, 'syn_pet')
            eval_model = os.path.join(trainer.results_folder, 'model.pt')
            trainer.evaluate(eval_model, synth_folder, synthesis=True, synthesis_folder = synth_folder, get_ROI_loss=True)
        else:
            eval_folder = os.path.join(trainer.results_folder, 'eval')
            eval_model = os.path.join(trainer.results_folder, 'model.pt')
            trainer.evaluate(eval_model, eval_folder)
    else:
        trainer.train()

if __name__ == "__main__":
    set_seed_everywhere(666)
    main()