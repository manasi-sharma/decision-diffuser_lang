import sys
sys.path.append('/iliad/u/manasis/decision-diffuser_lang/code/')

import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output

from diffuser.datasets.diffusionpolicy_datasets.base_dataset import BaseLowdimDataset
from diffuser.datasets.diffusionpolicy_datasets.kitchen_mjl_lowdim_dataset import KitchenMjlLowdimDataset
from diffuser.datasets.diffusionpolicy_datasets.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from diffuser.datasets.diffusionpolicy_datasets.v0 import KitchenAllV0

import re

from time import time

from voltron import instantiate_extractor, load

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()

    """Additional dataset configuration / langauge stuff"""
    cfg_valdataloader = {'batch_size': 256, 'num_workers': 1, 'persistent_workers': False, 'pin_memory': True, 'shuffle': False}
    cfg_task_dataset = {'abs_action': True, 'dataset_dir': 'data/kitchen/kitchen_demos_multitask', 'horizon': 16, 'pad_after': 7, 
                        'pad_before': 1, 'robot_noise_ratio': 0.1, 'seed': 42, 'val_ratio': 0.02}

    norm_dataset: BaseLowdimDataset
    #dataset = hydra.utils.instantiate(cfg_task_dataset) #cfg.task.dataset)
    norm_dataset = KitchenMjlLowdimDataset(**cfg_task_dataset)
    normalizer = norm_dataset.get_normalizer()
    """dataloader = cycle(DataLoader(self.dataset, **cfg_valdataloader)) #**cfg.dataloader)"""
    
    # phrase to sentence converter
    p_to_s = {
        'kettle': 'Move the kettle to the top burner',
        #'Kettle': 'Move the kettle to the top burner',
        'bottom burner': 'Turn the oven knob that activates the bottom burner', 
        #'BottomBurner': 'Turn the oven knob that activates the bottom burner', 
        'hinge cabinet': 'Open the hinge cabinet',
        #'HingeCabinet': 'Open the hinge cabinet',
        'slide cabinet': 'Open the slide cabinet',
        #'SlideCabinet': 'Open the slide cabinet',
        'light switch': 'Turn on the light switch',
        #'Light': 'Turn on the light switch',
        'top burner': 'Turn the oven knob that activates the top burner',
        #'TopBurner': 'Turn the oven knob that activates the top burner',
        'microwave': 'Open the microwave door',
        #'Microwave': 'Open the microwave door',
    }

    # Loading in Language Encoder
    vcond, preprocess = load("v-cond", device="cuda", freeze=True)
    vector_extractor = instantiate_extractor(vcond)()
    """End"""

    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer) #, train_or_val='val')
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    num_eval = 10
    device = Config.device

    # use abs_action=True
    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]

    #env_list = [KitchenLowdimWrapper(KitchenAllV0(use_abs_action=True)) for _ in range(num_eval)]
    #import pdb;pdb.set_trace()
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    # language list of tasks
    #list_tasks = ['Kitchen', 'Microwave', 'Kettle', 'BottomBurner', 'Light']
    
    # Setting up the language returns
    returns = []
    for i in range(num_eval):
        list_tasks = env_list[i].env.tasks_to_complete
        subtasks_sentence_list = [p_to_s[subtask] for subtask in list_tasks]
        subtasks_sentence = ', and '.join(subtasks_sentence_list).lower().capitalize()
        #import pdb;pdb.set_trace()
        multimodal_embeddings = vcond(subtasks_sentence, mode="multimodal")
        representation = vector_extractor(multimodal_embeddings.cpu())
        returns.append(representation)
    returns = torch.cat(returns)
    returns = to_device(returns, device)
    returns=None
    #returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]

    while sum(dones) <  num_eval:
        #t1 = time()
        
        #obs = dataset.normalizer.normalize({'obs': obs})
        obs = normalizer['obs'].normalize(obs)
        #obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns, verbose=False)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        #action = dataset.normalizer.unnormalize(action, 'actions')
        action = normalizer['action'].unnormalize(action)

        samples = to_np(samples)
        action = to_np(action)

        """if t == 0:
            normed_observations = samples[:, :, :]
            #observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            #observations = dataset.normalizer['obs'].unnormalize(observations)
            savepath = os.path.join('images', 'sample-planned.png')
            renderer.composite(savepath, observations)"""

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            print("reward: ", this_reward)
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward
        
        #print("\n\nTime for 1 eval run: ", time()-t1)
        
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    print("\n\nFinal t: ", t)

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    """renderer.composite(savepath, recorded_obs)"""
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})

if __name__ == "__main__":
    evaluate()