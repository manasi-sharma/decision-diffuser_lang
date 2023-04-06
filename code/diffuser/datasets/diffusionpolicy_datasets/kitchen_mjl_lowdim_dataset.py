from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from diffuser.datasets.diffusionpolicy_datasets.pytorch_util import dict_apply
from diffuser.datasets.diffusionpolicy_datasets.replay_buffer import ReplayBuffer
from diffuser.datasets.diffusionpolicy_datasets.sampler import SequenceSampler, get_val_mask
from diffuser.datasets.diffusionpolicy_datasets.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffuser.datasets.diffusionpolicy_datasets.base_dataset import BaseLowdimDataset
from diffuser.datasets.diffusionpolicy_datasets.kitchen_util import parse_mjl_logs

import re

from voltron import instantiate_extractor, load

class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        # Loading in Language Encoder
        vcond, preprocess = load("v-cond", device="cuda", freeze=True)
        vector_extractor = instantiate_extractor(vcond)()

        # phrase to sentence converter
        p_to_s = {
            'kettle': 'Move the kettle to the top burner',
            'bottomknob': 'Turn the oven knob that activates the bottom burner', 
            'hinge': 'Open the hinge cabinet',
            'slide': 'Open the slide cabinet',
            'switch': 'Turn on the light switch',
            'topknob': 'Turn the oven knob that activates the top burner',
            'microwave': 'Open the microwave door',
        }

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = np.concatenate([
                    qpos[:,:9],
                    qpos[:,-21:],
                    np.zeros((len(qpos),30),dtype=np.float32)
                ], axis=-1)
                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1., high=1., size=(obs.shape[0], 30))
                    obs[:,:30] += noise
                
                # loading in language
                found_1 = re.search('friday_(.+?)/', data['logName'])
                found_2 = re.search('postcorl_(.+?)/', data['logName'])
                if found_1:
                    lang = found_1.group(1)
                if found_2:
                    lang = found_2.group(1)

                # Phrase to full sentence
                subtasks = lang.split('_')
                subtasks_sentence_list = [p_to_s[subtask] for subtask in subtasks]
                subtasks_sentence = ', and '.join(subtasks_sentence_list).lower().capitalize()
                
                # Encoding in language model
                multimodal_embeddings = vcond(subtasks_sentence, mode="multimodal")
                representation = vector_extractor(multimodal_embeddings.cpu())
                lang_repr_indv = representation.detach().numpy()
                lang_repr = np.repeat(lang_repr_indv, obs.shape[0], axis=0)

                episode = {
                    'obs': obs,
                    'action': data['ctrl'].astype(np.float32),
                    'lang': lang_repr
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import pdb;pdb.set_trace()
        sample = self.sampler.sample_sequence(idx)
        import pdb;pdb.set_trace()
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
