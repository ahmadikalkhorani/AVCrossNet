This is the official pytorch implementation of 
# [AV-CrossNet: an Audiovisual Complex Spectral Mapping Network for Speech Separation By Leveraging Narrow- and Cross-Band Modeling](https://arxiv.org/abs/2406.11619).

#### Vahid Ahmadi Kalkhorani, Cheng Yu, Anurag Kumar, Ke Tan, Buye Xu, DeLiang Wang


## abstract 
Adding visual cues to audio-based speech separation can improve separation performance. This paper introduces AV-CrossNet, an audiovisual (AV) system for speech enhancement, target speaker extraction, and multi-talker speaker separation. AV-CrossNet is extended from the CrossNet architecture, which is a recently proposed network that performs complex spectral mapping for speech separation by leveraging global attention and positional encoding. To effectively utilize visual cues, the proposed system incorporates pre-extracted visual embeddings and employs a visual encoder comprising temporal convolutional layers. Audio and visual features are fused in an early fusion layer before feeding to AV-CrossNet blocks. We evaluate AV-CrossNet on multiple datasets, including LRS, VoxCeleb, and COG-MHEAR challenge. Evaluation results demonstrate that AV-CrossNet advances the state-of-the-art performance in all audiovisual tasks, even on untrained and mismatched datasets.



[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.11619) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ahmadikalkhorani.AVCrossNet)





# Pre-requisites
## creating a conda environment
```bash create_env.sh```

# Audiovisual Speaker Separation

## Training
```bash
conda activate avcrossnet

B = 4 # batch size

LOCAL_DATASET_PATH=/path/to/dataset/directory
ds=LRS2 # or LRS3/VoxCeleb2
exp_name=my_experiment_name

srun python -u AVSSTrainer.py fit \
    --config=configs/datasets/avss.yaml \
    --config=configs/AVCrossNet.yaml \
    --config=configs/loss/loss_hybrid.yaml \
    --config=configs/lr_scheduler/WarmupReduceLROnPlateau.yaml \
    --config=configs/loss/pit_false.yaml \
    --data.dataset_dir=$LOCAL_DATASET_PATH/AVSS/${ds}/audio_mixture/ \
    --data.config_dir=$LOCAL_DATASET_PATH/AVSS/${ds}/audio_mixture/ \
    --model.dataset_name=${ds} \
    --data.audio_time_len=[2.0,2.0,2.0] \
    --data.num_workers=8 \
    --data.visual_type="embedding" \
    --data.batch_size=[4,4] \
    --data.audio_only=False \
    --model.stft.n_fft=512 \
    --model.stft.n_hop=256 \
    --model.arch.num_freqs=257 \
    --model.arch.dim_input=2 \
    --model.arch.dim_output=4 \
    --model.sample_rate=16000 \
    --model.exp_name=$exp_name \
    --model.channels=[0] \
    --model.compile=False \
    --model.arch.positional_encoding=True \
    --model.arch.positional_encoding_type="random_chunk" \
    --trainer.precision="16-mixed" \
    --trainer.devices=-1 \
    --trainer.accelerator="gpu" \
    --trainer.max_epochs=200 \
    --model.metrics=[SNR,SDR,SI_SDR,WB_PESQ,eSTOI,STOI] \
```

## Test 

```bash
conda activate avcrossnet

best_ckpt=$(python scripts/best_ckpt.py --path path/to/current/exp/log/folder) 

cfg_path="$(dirname $(dirname "$best_ckpt"))/config.yaml"


python -u AVSSTrainer.py test \
    --config=$cfg_path \
    --model.metrics=[SDR,SI_SDR,NB_PESQ,eSTOI] \
    --model.write_examples=20 \
    --ckpt_path=$best_ckpt \

```

# Audiovisual Speaker Extraction

## Training
```bash
srun python -u AVSETrainer.py fit \
    --config=configs/AVCrossNet.yaml \
    --config=configs/voxceleb2.yaml \
    --config=configs/loss/loss_hybrid.yaml \
    --config=configs/lr_scheduler/ExponentialLR.yaml \
    --data.audio_time_len=[2.0,2.0,null] \
    --data.visual_type="embedding" \
    --model.visual_type="embedding" \
    --data.batch_size=[3,3] \
    --data.audio_only=False \
    --data.num_workers=5 \
    --model.stft.n_fft=512 \
    --model.stft.n_hop=256 \
    --model.arch.num_freqs=257 \
    --model.arch.dim_input=2 \
    --model.arch.dim_output=2 \
    --model.sample_rate=16000 \
    --model.exp_name=$SLURM_JOB_NAME \
    --model.channels=[0] \
    --model.compile=False \
    --model.arch.positional_encoding=True \
    --model.arch.positional_encoding_type="random_chunk" \
    --trainer.precision="16-mixed" \
    --trainer.devices=-1 \
    --trainer.max_epochs=200 \
    --trainer.accelerator="gpu" \
    --ckpt_path=path/to/last/ckpt.ckpt \
```
## Test
```bash


best_ckpt=$(python scripts/best_ckpt.py --path path/to/current/exp/log/folder) 

cfg_path="$(dirname $(dirname "$best_ckpt"))/config.yaml"


python -u AVSETrainer.py test \
    --config=$cfg_path \
    --model.metrics=[SDR,SI_SDR,NB_PESQ,eSTOI] \
    --model.write_examples=20 \
    --ckpt_path=$best_ckpt \
```
# Acknowledgements

This is repository is mainly adopted from [NBSS](https://github.com/Audio-WestlakeU/NBSS). We thank the authors for their great code. 

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{kalkhorani2024av,
  title={{AV-CrossNet}: an Audiovisual Complex Spectral Mapping Network for Speech Separation By Leveraging Narrow-and Cross-Band Modeling},
  author={Kalkhorani, Vahid Ahmadi and Yu, Cheng and Kumar, Anurag and Tan, Ke and Xu, Buye and Wang, De Liang},
  journal={arXiv:2406.11619},
  year={2024}
}
```