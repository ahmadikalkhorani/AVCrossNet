# -*- coding: utf-8 -*-
'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''
import sys 
import os
import math
import logging
import numpy as np
import soundfile
import torch

from soundfile import SoundFile
from scipy.signal import convolve
from torchmetrics.audio import SignalNoiseRatio
# -*- coding: utf-8 -*-
'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

import scipy
import scipy.io
import random
from os.path import *
from pathlib import Path
from typing import *
import json
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy.signal import convolve, resample
from torch.utils.data import DataLoader, Dataset
import torchaudio
from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.rand import randfloat, randint, randnormal, randchoice
import torchvision 
from torchvision.io import read_video 
from torchaudio import load as read_audio
import math 
from torch.utils.data.distributed import DistributedSampler, T_co
from torchmetrics.audio import SignalNoiseRatio as SNR








# SPEECH_FILTER = scipy.io.loadmat(
#     os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, 'configs/datasets/speech_weight.mat'),
#     squeeze_me=True,
# )
# SPEECH_FILTER = np.array(SPEECH_FILTER["filt"])

def speechweighted_snr(target, noise):
    """Apply speech weighting filter to signals and get SNR."""
    target_filt = scipy.signal.convolve(
        target, SPEECH_FILTER, mode="full", method="fft"
    )
    noise_filt = scipy.signal.convolve(noise, SPEECH_FILTER, mode="full", method="fft")

    # rms of the target after speech weighted filter
    targ_rms = np.sqrt(np.mean(target_filt ** 2))

    # rms of the noise after speech weighted filter
    noise_rms = np.sqrt(np.mean(noise_filt ** 2))

    if noise_rms==0:
        return np.Inf

    sw_snr = np.divide(targ_rms, noise_rms)
    return sw_snr


def sum_signals(signals):
    """Return sum of a list of signals.

    Signals are stored as a list of ndarrays whose size can vary in the first
    dimension, i.e., so can sum mono or stereo signals etc.
    Shorter signals are zero padded to the length of the longest.

    Args:
        signals (list): List of signals stored as ndarrays

    Returns:
        ndarray: The sum of the signals

    """
    max_length = max(x.shape[0] for x in signals)
    return sum(pad(x, max_length) for x in signals)


def pad(signal, length):
    """Zero pad signal to required length.

    Assumes required length is not less than input length.
    """
    assert length >= signal.shape[0]
    return np.pad(
        signal, [(0, length - signal.shape[0])] + [(0, 0)] * (len(signal.shape) - 1)
    )

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Renderer:
    """
    SceneGenerator of AVSS1 training and development sets. The render() function generates all simulated signals for each
    scene given the parameters specified in the metadata/scenes.train.json or metadata/scenes.dev.json file.
    """

    def __init__(
        self,
        num_channels=1,
        fs=16000,
        ramp_duration=0.5,
        tail_duration=0.2,
        test_nbits=16,
    ):

        self.fs = fs
        self.ramp_duration = ramp_duration
        self.n_tail = int(tail_duration * fs)
        self.test_nbits = test_nbits
        self.floating_point = False

        self.channels = list(range(num_channels))


    def apply_ramp(self, x, dur):
        """Apply half cosine ramp into and out of signal

        dur - ramp duration in seconds
        """
        ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(self.fs * dur)))
        ramp = (ramp + 1) / 2
        y = np.array(x)
        y[0 : len(ramp)] *= ramp
        y[-len(ramp) :] *= ramp[::-1]
        return y

    def compute_snr(self, target, noise):
        """Return the SNR.
        Take the overlapping segment of the noise and get the speech-weighted
        better ear SNR. (Note, SNR is a ratio -- not in dB.)
        """
        segment_target = target
        segment_noise = noise
        assert len(segment_target) == len(segment_noise)

        snr = speechweighted_snr(segment_target, segment_noise)

        return snr

    def render(
        self,
        spk1,
        spk2,
        snr_dB,
    ):
        source = [spk1, spk2]
        snrs = snr_dB
        

        activlev_scales = [torch.clamp(torch.sqrt(torch.mean(s**2)), min = 1E-5) for s in source]  # If no activlev file
        scaled_sources = [s / torch.sqrt(scale) * 10 ** (x/20) for s, scale, x in zip(source, activlev_scales, snrs)]

        sources_np = torch.stack(scaled_sources, axis=0)
        mix_np = torch.sum(sources_np, axis=0)

        gain = np.max([1., torch.max(torch.abs(mix_np)), torch.max(torch.abs(sources_np))]) / 0.9
        mix = mix_np / gain
        sources_np_max = sources_np / gain
        
        spk1, spk2 = sources_np_max[0], sources_np_max[1]
        

        return spk1, spk2, mix

        

def check_scene_exists(scene, output_path):
    """Checks correct dataset directory for full set of pre-existing files.

    Args:
        scene (dict): dictionary defining the scene to be generated.

    Returns:
        status: boolean value indicating whether scene signals exist
            or do not exist.

    """

    pattern = f"{output_path}/{scene['scene']}"
    files_to_check = [
        f"{pattern}_mixed.wav",
        f"{pattern}_target.wav",
        f"{pattern}_interferer_1.wav",
    ]

    scene_exists = True
    for filename in files_to_check:
        scene_exists = scene_exists and os.path.exists(filename)
    return scene_exists








def resample_video(v, orig_fps, new_fps):
    N_frames = int(v.shape[0] / orig_fps * new_fps) 
    
    if orig_fps > new_fps:
        idx = torch.randperm(v.shape[0])[:N_frames]
        v = v[idx]
        
    if orig_fps < new_fps:
        
        # we need to add n new frames: 
        n = N_frames - v.shape[0] 
        
        idx = torch.randint(low = 0, high = v.shape[0], size = (n,)) 
        
        # concatenate the nex idx to the previois idx
        idx = torch.cat([torch.arange(v.shape[0]), idx]).sort()[0]
        
        v = v[idx]
        
    return v

class AVSSDataset(Dataset):
    """The AVSS! dataset"""

    def __init__(
        self,
        dataset_dir: str,
        dataset: str = "train",
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
        audio_only: bool = True,
        fps: int = 25,
        config_dir: str = "configs/datasets",
        snr_sch: bool = False, # do not change this, it should be False, SNRShceduler callback will change it 
        dynamic_mix: bool = False,
        visual_type: str = "embedding",
        num_spks: int = 2,
        scene_classification: bool = False, 
        mixed_noise: bool = False, 
        batch_size : int = 4, 
        speech_noise_probability: float = 0.7,
    ) -> None:
        """The AVSS! dataset

        Args:
            dataset_dir: a dir contains [wav8k]
            dataset: tr, cv, tt
            target: anechoic or reverb
            version: min or max
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert sample_rate in [8000, 16000], sample_rate
        assert dataset in ['train', 'val', 'test'], dataset
        # assert visual_type in ["image", "embedding_avsr", "embedding_autovsr"], visual_type
        self.snr_sch = snr_sch
        self.noise_scale = None
        self.DM = dynamic_mix
        self.visual_type = visual_type
        self.num_spks = num_spks
        self.mixed_noise = mixed_noise 
        self.scene_classification = scene_classification
        self.speech_noise_probability = speech_noise_probability

        self.dataset_dir = str(Path(dataset_dir).expanduser())
        self.wav_dir = Path(self.dataset_dir) 
        
        self.batch_size = batch_size
        
        self.name = config_dir.split("/")[-2]
        
        with open(os.path.join(config_dir, "cfg.json"), "r") as f:
            print("loading: ", os.path.join(config_dir, "cfg.json"))
            scenes = json.load(f)
            print(scenes.keys())
            scenes = scenes[dataset]
        
        self.scenes = scenes
        self.noise_scenes = scenes

        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate 
        self.fps = fps
        
        self.av = not audio_only
        
        self.seed = 0
        
        self.renderer = Renderer(num_channels=1, fs=sample_rate, ramp_duration=0.5, tail_duration=0.2, test_nbits=16)
        
        self.g = torch.Generator()
        
        print("visual type:", self.visual_type, "scenes len:", len(self.scenes), "noise scenes len:", len(self.noise_scenes) )
    
    def read(self, wav_path: str, N: int, start: int = None, g: torch.Generator = None) -> Tuple[torch.Tensor, int]:
        
        '''
        Read the wav file and return the tensor of the audio signal
        wav_path: str, the path to the wav file
        N: int, the number of samples to read, if None, read the whole file
        g: torch.Generator, the random generator
        '''
        
        # print("wav_path : ", wav_path)
        
        clean, samplerate = sf.read(wav_path, dtype='float32')
        assert len(clean.shape) == 1, "clean speech should be single channel"
        # resample if necessary
        if self.sample_rate != None and samplerate != self.sample_rate:
            re_len = int(clean.shape[0] * self.sample_rate / samplerate)
            clean = resample(clean, re_len)
        
        wav = torch.from_numpy(clean).reshape(1, -1)
        
        # remove zeros at the end of file: 
        wav = wav[:, :torch.max(torch.nonzero(wav) + 1)]
        
        if N is not None:
            if wav.shape[-1] > N:
                start = randint(g, 0, wav.shape[-1] - N) if start is None else start
                wav = wav[..., start:start + N]  
            elif wav.shape[-1] < N:
                wav = torch.cat([wav]*int(N/wav.shape[-1] + 1), dim = -1).reshape(1, -1)[..., :N]
        
        return wav, samplerate, start

    def read_embedding(self, path, t1: float, n_frames: int): 
        n1 = int(t1 * self.fps) if t1 is not None else 0
        n2 = n1 + n_frames if n_frames is not None else None
        path = path.replace("_face_", "_lip_").replace(".png", ".npy")
        
        v = np.load(path, allow_pickle=True)[n1:n2] # frame x 512 
        if n_frames is None:
            return torch.from_numpy(v).unsqueeze(0).float()  
        if v.shape[0] < n_frames:
            v = np.concatenate([v, np.zeros((n_frames - v.shape[0], *v.shape[1:] ))], axis = 0)
        elif v.shape[0] > n_frames:
            v = v[:n_frames]
        return torch.from_numpy(v).unsqueeze(0).float() 

    def read_face(self, path, t1: float, n_frames: int):
        
        if "embedding" in self.visual_type.lower():
            return self.read_embedding(path, t1, n_frames) 

        if ".png" in path or ".jpg" in path:
            raise NotImplementedError("Image type not implemented yet")
            n1 = int(t1 * self.fps)
            n2 = int(t2 * self.fps) if t2 is not None else None 
            v = torchvision.io.read_image(path)
            
            v = torch.stack(torch.split(v, v.shape[1], dim = -1), dim = 0)
            v = v.float()
            v = (v - v.min()) / (v.max() - v.min()) 
            v = v[n1:n2]
            return v.unsqueeze(0)
        
        v, a, p = read_video(path, output_format="TCHW", pts_unit="sec", start_pts=t1, end_pts=t2) 
        
        if p["video_fps"] != self.fps:
            v = resample_video(v, orig_fps=p["video_fps"], new_fps=self.fps) 
        
        v = v.float() 
        
        v = (v - v.min()) / (v.max() - v.min())
        
        
        if v.shape[0] < n_frames:
            v = torch.cat([v, torch.zeros((n_frames - v.shape[0], *v.shape[1:] ))], dim = 0)
        elif v.shape[0] > n_frames:
            v = v[:n_frames]
            
            
        return v.unsqueeze(0)
    
    def change_noise_scale(self, snr_bounds):
        self.noise_scale = snr_bounds
        self.snr_sch = True
        print(f"SNR bounds changed to {snr_bounds}")
        
    
    def __len__(self):
        if (self.dataset == 'train')&(self.DM | self.snr_sch):
            gpus = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            return gpus * self.batch_size * 20000 # (batch/2) x gpu x 5000
        return self.speech_num() * self.num_spks # since we need to return both speakers

    def speech_num(self):
        return len(self.scenes)
    
    def read_noise(self, wav_path: str, N: int, g: torch.Generator) -> torch.Tensor:
        noise, sr = self.read(wav_path, )       
        
        if noise.shape[-1] > N:
            start = randint(g, 0, noise.shape[-1] - N)
            noise = noise[..., start:start + N] 
        
        elif noise.shape[-1] < N:
            noise = torch.cat([noise]*int(N/noise.shape[-1] + 1), dim = -1).reshape(1, -1)[..., :N]
            
        return noise
    
    def read_spk(self, path, N: int, g: torch.Generator, start: int = None, ):
        spk_wav_path = path
        spk_vid_path = path.replace("_spk", "_face_spk").replace(".wav", ".png")
        
        wav, sr, start = self.read(spk_wav_path, N = N, start = start, g = g, ) 
            
        v = self.read_face(spk_vid_path, t1 = start / sr if start is not None else None, n_frames = int(wav.shape[-1]/self.sample_rate * self.fps))
        
        return wav, v, start
        

    def __getitem__(self, idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:  # type: ignore
        """returns the indexed item

        Args:
            index: index

        Returns:
            Tensor: xm of shape [channel, time]
            Tensor: ys of shape [spk, channel, time]
            dict: paras used
        """
        
        if isinstance(idx, int):
            idx = {'speech_index': idx, 'seed': torch.randint(high=9999999999, size=(1,)).item(), 'dynamic_seed': torch.randint(high=2**31-1, size=(1,)).item()}
        
        g = self.g
        if self.dataset != 'train':
            g.manual_seed(idx['seed'])
        else: 
            g.manual_seed(idx['dynamic_seed'])  
            np.random.seed(idx['dynamic_seed'])
            # g.manual_seed( np.random.randint(0, 9999999999) )
        
        index = idx['speech_index']
        
      
        try: 
        # if True:
            
            
            if self.dynamic_mix: 
                scene = self.scenes[ randint(g, 0, len(self.scenes)) ]
            else:
                scene = self.scenes[index % len(self.scenes)]
            
            frames = None
            if self.audio_time_len:
                frames = int(self.sample_rate * self.audio_time_len)

            
            
            

            spk2_scene = {} # for logging purposes in val/test mode
            if self.snr_sch or self.DM:
                
                i = 1 if randfloat(g = g, low = 0, high = 1) > 0.5 else 2
                spk1_wav_path = os.path.join(self.wav_dir, scene["dataset"],  scene["scene"] + f"_spk{i}.wav")
                wav1, v1, start = self.read_spk(spk1_wav_path, N = frames, g = g)
                while(torch.sqrt(torch.mean(wav1**2)) < 1E-5):
                    wav1, v1, start = self.read_spk(spk1_wav_path, N = frames, g = g)
                
                wav2 = torch.zeros_like(wav1)
                while(torch.sqrt(torch.mean(wav2**2)) < 1E-5):
                
                    spk2_scene = self.scenes[randint(g, 0, len(self.scenes))] 
                    i = 1 if randfloat(g = g, low = 0, high = 1) > 0.5 else 2
                    spk2_wav_path = os.path.join(self.wav_dir, spk2_scene["dataset"],  spk2_scene["scene"] + f"_spk{i}.wav")
                    wav2, v2, _ = self.read_spk(spk2_wav_path, N = frames, g = g)
                    
                snr1 = round(randfloat(g, 0, 5), 4)
                snr = [snr1, -snr1]
                
                wav1, wav2, mix = self.renderer.render(
                    spk1 = wav1, 
                    spk2 = wav2,
                    snr_dB = snr, 
                )
            
          

            else: 
                snr = None 
                
                start = 0 if self.dataset in ["val", "test"] else None
                
                
                    
                i1 = 1 if index % 2 == 0 else 2 
                i2 = [None, 2, 1][i1]
                
                spk1_wav_path = os.path.join(self.wav_dir, scene["dataset"],  scene["scene"] + f"_spk{i1}.wav")
                wav1, v1, start = self.read_spk(spk1_wav_path, N = frames, g = g, start = start)
            
                spk2_wav_path = os.path.join(self.wav_dir, scene["dataset"],  scene["scene"] + f"_spk{i2}.wav")
                wav2, v2, _ = self.read_spk(spk2_wav_path, N = wav1.shape[-1], g = g, start = start)
                
                mix_path = os.path.join(self.wav_dir, scene["dataset"],  scene["scene"] + "_mixed.wav")
                mix, sr, _ = self.read(mix_path, N = wav1.shape[-1], start = start)
                
                
         
                
            
                
            
            if mix.ndim == 1:
                mix = mix.reshape(1, -1)
                wav1 = wav1.reshape(1, -1)
                wav2 = wav2.reshape(1, -1)
            
            if wav1.ndim == 2:
                wav1 = wav1.unsqueeze(1) 
                wav2 = wav2.unsqueeze(1)
                
           
            max_amp = max([torch.max(torch.abs(mix)), torch.max(torch.abs(wav1)), torch.max(torch.abs(wav2)), 1E-5])
            amp_scaling = 0.9 / max_amp
            mix *= amp_scaling
            wav1 *= amp_scaling
            wav2 *= amp_scaling
            
            target = torch.cat([wav1, wav2], dim = 0)
            
            
            
            
            
            paras = {
                **idx,
                'index': index,
                'seed': self.seed,
                'sample_rate': self.sample_rate,
                'dataset': self.dataset,
                'audio_time_len': self.audio_time_len,
                "num_samples": mix.shape[-1], 
                "spk1_wav_path": spk1_wav_path,
                "spk2_wav_path": spk2_wav_path,
                "dynamic_mixing": self.DM,
                "snr_sch": self.snr_sch,
                "SNR": snr,
                **scene, 
                
            }
            
            if index == 0:
                print(paras)
                
            

            x, y = torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)
            

            if self.av:
                v = torch.cat([v1, v2], dim = 0)        
                return x, y[:1], v[:1], paras
    
            
            return x, y, paras

        except Exception as e:
            print(f'error in reading {index}: {scene}')
            return self.__getitem__((index + 1)%self.__len__())
    
    



class SS_SemiOnlineSampler(DistributedSampler[T_co]):
    r"""Sampler for SS_SemiOnlineDataset for single GPU and multi GPU (or Distributed) cases.
    If shuffle == True, the speech pair sequence and seed changes along epochs, else the speech pair and seed won't change along epochs
    If shuffle_rir == True, the rir will be shuffled, otherwise not

    No matter what is ``shuffle`` or ``shuffle_rir``, the speech sequence, rir sequence, seed generated for dataset are all deterministic.
    They all determined by the parameter ``seed`` and ``epoch`` (for shuffle == True)
    """

    def __init__(self,
                 dataset: AVSSDataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        try:
            super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        except:
            # if error raises, it is running on single GPU
            # thus, set num_replicas=1, rank=0
            super().__init__(dataset, 1, 0, shuffle, seed, drop_last)
        self.last_epoch = -1

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            speech_indices = torch.randperm(self.dataset.speech_num(), generator=g).tolist()  # type: ignore
          

            if self.last_epoch >= self.epoch:
                import warnings
                if self.epoch != 0:
                    warnings.warn('the epoch value doesn\'t update when shuffle is true, the training data and sequence won\'t change along with different epochs')
            else:
                self.last_epoch = self.epoch
        else:
            g = torch.Generator()
            g.manual_seed(self.seed)
            speech_indices = list(range(len(self.dataset)))  # type: ignore
        
        # make rir_indices and speech_indices have the same length as the dataset
        if len(speech_indices) > len(self.dataset):  # type: ignore
            speech_indices = speech_indices[:len(self.dataset)]  # type: ignore

        # construct indices
        indices = []
        for i, sidx in enumerate(speech_indices):
            seed = torch.randint(high=9999999999, size=(1,), generator=g)[0].item()
            indices.append({'speech_index': sidx, 'seed': seed, 'dynamic_seed': i + self.epoch + torch.randint(high=2**31 - (i + self.epoch), size=(1,))[0].item() })

        # drop last
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)  # type: ignore

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class AVSSDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_dir: str,
        sample_rate: int = 16000,
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],  # audio_time_len (seconds) for training, val, test.
        batch_size: List[int] = [1, 1],
        test_set: str = 'test',  # the dataset to test: train, val, test
        num_workers: int = 5,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int] = [None, 2, 3],  # random seeds for train, val and test sets
        pin_memory: bool = True,
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
        audio_only: bool = True,
        dynamic_mix: bool = True,
        config_dir: str = "configs/datasets",
        visual_type: str = "image",
        num_spks: int = 2,
        mixed_noise: bool = False, 
        scene_classification: bool = False, 
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.audio_time_len = audio_time_len
        self.persistent_workers = persistent_workers
        self.test_set = test_set
        self.ao = audio_only
        self.DM = dynamic_mix
        self.visual_type = visual_type
        self.num_spks = num_spks
        self.scene_classification = scene_classification 
        self.mixed_noise = mixed_noise
        
  

        rank_zero_info(f'dataset: AVSS!, datasets for train/valid/test:  {sample_rate}, time length: {audio_time_len}')
        assert audio_time_len[2] == None, audio_time_len

        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]
        rank_zero_info(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test={self.batch_size_test}')
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"

        self.num_workers = num_workers

        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.seeds = []
        seeds[0] = None
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))
        
        print(f"Dataset seeds: {self.seeds}")

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        self.config_dir = config_dir

    def setup(self, stage=None):
        if stage is not None and stage == 'test':
            audio_time_len = None
        else:
            audio_time_len = self.audio_time_len

        self.train = AVSSDataset(
            dataset_dir=self.dataset_dir,
            dataset='train',
            audio_time_len=self.audio_time_len[0],
            sample_rate=self.sample_rate,
            audio_only=self.ao,
            config_dir = self.config_dir,
            dynamic_mix=self.DM,
            visual_type = self.visual_type,
            num_spks = self.num_spks,
            scene_classification = self.scene_classification, 
            mixed_noise= self.mixed_noise,
            batch_size=self.batch_size,
            
        )
        self.val = AVSSDataset(
            dataset_dir=self.dataset_dir,
            dataset='val',
            audio_time_len=self.audio_time_len[1], # same as CTCNet
            sample_rate=self.sample_rate,
            audio_only=self.ao,
            config_dir = self.config_dir,
            dynamic_mix=False,
            visual_type = self.visual_type,
            num_spks = self.num_spks,
            scene_classification = self.scene_classification, 
            mixed_noise= self.mixed_noise,
            batch_size=self.batch_size_val,
        )
        self.test = AVSSDataset(
            dataset_dir=self.dataset_dir,
            dataset='test',
            audio_time_len=self.audio_time_len[2], # same as CTCNet
            sample_rate=self.sample_rate,
            audio_only=self.ao,
            config_dir = self.config_dir,
            dynamic_mix=False, 
            snr_sch=False, 
            visual_type = self.visual_type,
            num_spks = self.num_spks,
            scene_classification = self.scene_classification, 
            mixed_noise= self.mixed_noise,
            batch_size=1,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            sampler=SS_SemiOnlineSampler(self.train, seed=self.seeds[0], shuffle=True),
            batch_size=self.batch_size,
            collate_fn=self.collate_func_train,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seeds[0] + worker_id),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            sampler=SS_SemiOnlineSampler(self.val, seed=self.seeds[1], shuffle=False),
            batch_size=self.batch_size_val,
            collate_fn=self.collate_func_val,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
            sampler=SS_SemiOnlineSampler(self.test, seed=self.seeds[2], shuffle=False),
            batch_size=self.batch_size_test,
            collate_fn=self.collate_func_test,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


if __name__ == '__main__':

    
    from argparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_argument("--dataset_dir", type=str, )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_time_len", type=float, default=1.0)
    parser.add_argument("--batch_size", type=list, default=[1,1])
    parser.add_argument("--dynamic_mix", type=bool, default=True)
    parser.add_argument("--visual_type", type=str, default="embedding")
    
    
    args = parser.parse_args()
    

    
    dm = AVSSDataModule(
        dataset_dir=args.dataset_dir,
        sample_rate=args.sample_rate,
        audio_time_len=[args.audio_time_len, args.audio_time_len, None],
        batch_size=args.batch_size,
        dynamic_mix=args.dynamic_mix,
        visual_type=args.visual_type,
        audio_only=False,
        scene_classification=False,    
        mixed_noise=True,
        config_dir=args.dataset_dir,
        pin_memory=False,
    )
    
    dm.setup()
    
    dl = dm.test_dataloader()
    
    batch = next(iter(dl))  
    
    x, y, v, p = batch
    print(f"x shape: {x.shape}, y shape: {y.shape}, v shape: {v.shape}, p: {p}")
    # print(f"SNR = {SNR()(x, y.squeeze(1)).item()}, ")
    # print(f"z = {z}")
    