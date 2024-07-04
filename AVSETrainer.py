from models.arch.visual_frontend import VisualFrontend
from models.utils.base_cli import BaseCLI
# import BaseCLI at the beginning

import os
from typing import *
from models.utils.asr_metric import AutoASR

import pytorch_lightning as pl
import torch
import torch.nn as nn
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn.functional import interpolate

import models.utils.general_steps as GS
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.arch.vtp import VTP_wrapper, CNN_3d_featextractor, VTP
import data_loaders
from utils import bcolors
from einops import rearrange

class TrainModule(pl.LightningModule):
    """Network Lightning Module, which controls the training, testing, and inference of given arch and io
    """
    name: str  # used by CLI for creating logging dir
    import_path: str = 'SharedTrainer.TrainModule'

    def __init__(
        self,
        arch: nn.Module,
        channels: List[int],
        ref_channel: int,
        stft: STFT = STFT(n_fft=256, n_hop=128, win_len=256),
        norm: Norm = Norm(mode='utterance'),
        loss: Loss = Loss(loss_func=neg_si_sdr, pit=True),
        optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {
            "lr": 0.001
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        val_metric: str = 'loss',
        write_examples: int = 200,
        ensemble: Union[int, str, List[str], Literal[None]] = None,
        compile: bool = False,
        exp_name: str = "exp",
        weights_ckpt: str = None,
        sample_rate: int = 8000,
        asr_model_name: str = "small",
        visual_type: str = "embedding",
        dataset_name: str = "test",
    ):
        """
        Args:
            exp_name: set exp_name to notag when debug things. Defaults to "exp".
            metrics: metrics used at test time. Defaults to ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'].
            write_examples: write how many examples at test.
        """

        super().__init__()

        args = locals().copy()  # capture the parameters passed to this function or their edited values

        self.sample_rate = sample_rate
      

        if compile != False:
            torch._dynamo.config.verbose = True
            assert Version(torch.__version__) >= Version('2.0.0'), f'compile only works for torch>=2.0: current version: {torch.__version__}'
            self.arch = torch.compile(arch)
        else:
            self.arch = arch

  
        self.channels = channels
        self.ref_channel = ref_channel
        self.stft = stft
        self.norm = norm
        self.loss = loss

        self.val_cpu_metric_input = []
        self.norm_if_exceed_1 = True
        self.name = type(arch).__name__

        # save other parameters to `self`
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)
        
        if weights_ckpt is not None:
            print("*"*50)
            self.load_weights(self, weights_ckpt)
        
        # asr metric 
        self.asr = None
        self.asr_model_name = asr_model_name
     
        self.visual_type = visual_type 
        if "embedding" not in self.visual_type:
            self.v_front = VisualFrontend()
        
        
    def load_weights(self, model, ckpt_path) :
        def load_pretrained_module(module, pretrained):
            # rename weights for removing _orig_mod in name
            if not self.compile:
                pretrained = {k.replace('_orig_mod.', '') : v for k, v in pretrained.items()}
          
            
            pretrained_dict = pretrained # speech brain pretrained model
            model_dict = module.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
            missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
            
            print(bcolors.WARNING, 'loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)) , bcolors.ENDC)
            print(bcolors.FAIL,'missed_params: \n', missed_params , bcolors.ENDC)
            # print('miss matched params:',missed_params)
            model_dict.update(pretrained_dict)
            module.load_state_dict(model_dict)
        
        pt = torch.load(ckpt_path, map_location=lambda storage, loc: storage )
        load_pretrained_module(model, pt["state_dict"])
    
        del pt

    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        GS.on_train_start(self=self, exp_name=self.exp_name, model_name=self.name, num_chns=max(self.channels) + 1, nfft=self.stft.n_fft, model_class_path=self.import_path)

    def forward(self, x: Tensor, v: Tensor = None) -> Tensor:
        """
        Args:
            x: [B,C,T]
            v: [B, T, C, H, W]

        Returns:
            Tuple[Tensor, Tensor]: ys_hat
        """
        
        v = v.squeeze(1)
        X, stft_paras = self.stft.stft(x[:, self.channels])  # [B,C,F,T], complex
        B, C, F, T = X.shape
        X, norm_paras = self.norm.norm(X, ref_channel=self.channels.index(self.ref_channel))
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C
        
        
        if "emb" not in self.visual_type:
            if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
                with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                    v = self.v_front(v)
            else: 
                v = self.v_front(v)
        
        # network process
        out = self.arch(X, v = v)
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk,2]
            
        
        
        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        

        # to time domain
        Yr_hat = self.norm.inorm(out, norm_paras)
        yr_hat = self.stft.istft(Yr_hat, stft_paras) 
        
        
        
        return {
            "preds": yr_hat, 
        }

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, ys, v, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        yr = ys[:, :, self.ref_channel, :]
        
        
        pred = self.forward(x, v)
        yr_hat = pred["preds"]
        
    

        # float32 loss calculation
        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=False, reduce_batch=True)  # convert to float32 to avoid numerical problem in loss calculation
        else:
            loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=False, reduce_batch=True)

        si_sdr_train = si_sdr(preds=yr_hat, target=yr).mean()
        self.log('train/' + self.loss.name, loss, batch_size=ys[0].shape[0], prog_bar=True)
        self.log('train/si_sdr', si_sdr_train, sync_dist=True, batch_size=ys.shape[0], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, ys, v, paras = batch
        yr = ys[:, :, self.ref_channel, :]

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            # 我也不知道为什么：self.forward放在autocast之后就会出问题，难道是因为lightning内部的GradScaler的原因？
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        # forward & loss
        pred = self.forward(x, v)
        yr_hat = pred["preds"]
        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True)
        
        # for tensorboard log
        self._mix, self._preds, self._target = x[0, self.ref_channel], yr_hat[0], yr[0]

        # metrics
        # sdr_val = sdr(yr_hat, yr).mean()
        si_sdr_val = si_sdr(preds=yr_hat, target=yr).mean()

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # logging
        self.log('val/' + self.loss.name, loss, sync_dist=True, batch_size=ys.shape[0])
        self.log('val/loss', loss, sync_dist=True, batch_size=ys.shape[0])
        # val_metric = {'loss': -loss, 'si_sdr': si_sdr_val, 'sdr': sdr_val}[self.val_metric]
        val_metric = {'loss': -loss, 'si_sdr': si_sdr_val,}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=ys.shape[0])  # log val/metric for checkpoint picking

        # always computes the sdr/sisdr for the comparison of different runs
        # self.log('val/sdr', sdr_val, sync_dist=True, batch_size=ys.shape[0])
        if self.loss.name != 'neg_si_sdr':
            # always computes the neg_si_sdr for the comparison of different runs in Tensorboard
            self.log('val/neg_si_sdr', -si_sdr_val, sync_dist=True, batch_size=ys.shape[0])

        # other heavy metrics: pesq
        sample_rate = paras[0]['sample_rate']
        yrs = [[
            ['nb_pesq'] if sample_rate == 8000 else ['nb_pesq', 'wb_pesq'],
            yr_hat.detach().cpu(),
            yr.detach().cpu(),
            None,
            sample_rate,
            'cpu',
        ]]
        self.val_cpu_metric_input += yrs

    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)

    def on_test_epoch_start(self):
        self.exp_save_path = os.path.join(self.trainer.logger.log_dir, self.dataset_name)
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []
        
        if ("WER" in self.metrics)&(self.asr is None):
            self.asr = AutoASR(sr = self.sample_rate, device = self.device, model_name = self.asr_model_name)

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path, dataset_name=self.dataset_name)


    def test_step(self, batch, batch_idx):
        x, ys, v, paras = batch
        yr = ys[:, :, self.ref_channel, :]
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        pred = self.forward(x, v)
        yr_hat = pred["preds"]
        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True)
        self.log('test/' + self.loss.name, loss, logger=False, batch_size=ys.shape[0])

        # write results & infos
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, self.loss.name: loss.item()}

        # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
        if self.loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

          # calculate metrics, input_metrics, improve_metrics on GPU
        if self.asr is not None:
            transcription = paras[0]['Transcription'] if "Transcription" in paras[0].keys() else paras[0]["spk1"]["Transcription"]
            if not isinstance(transcription, list):
                transcription = [transcription] 
            if len(transcription) != yr_hat.shape[-2]:
                transcription = transcription * yr_hat.shape[-2]

            metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref.expand_as(yr[0]), sample_rate, device_only='gpu', asr = self.asr, target_transcript = transcription)
        else:
            metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref.expand_as(yr[0]), sample_rate, device_only='gpu',)
       
       
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        self.cpu_metric_input.append((self.metrics, yr_hat[0].detach().cpu(), yr[0].detach().cpu(), x_ref.expand_as(yr[0]).detach().cpu(), sample_rate, 'cpu'))

        # write examples
        if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
            GS.test_setp_write_example(
                self=self,
                xr=x[:, self.ref_channel],
                yr=yr,
                yr_hat=yr_hat,
                sample_rate=sample_rate,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname,
                exp_save_path=self.exp_save_path,
            )

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # return metrics, which will be collected, saved in test_epoch_end
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        self.results.append(result_dict)
        return result_dict

    def predict_step(self, batch: Union[Tensor, Tuple[Tensor, Tensor, Dict]], batch_idx: Optional[int] = None, dataloader_idx: Optional[int] = None) -> Tensor:
        """predict step on self.device, could be called dirctly or by PytorchLightning automatically using predict dataset
        Args:
            batch: x or (x, ys, paras). shape of x [B, C, T]

        Returns:
            Tensor: ys_hat, shape [B, Spk, T]
        """
        if isinstance(batch, Tensor):
            x, ys = batch, None
            yr = None
        else:
            x, ys, v, paras = batch
            yr = ys[:, :, self.ref_channel, :] if ys[0] is not None else None

        # forward & loss
        pred = self.forward(x, v)
        yr_hat = pred["preds"]

        if self.loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        if yr is not None:  # reorder yr_hat if given yr
            _, perms = pit(preds=yr_hat, target=yr, metric_func=si_sdr, eval_func='max')
            yr_hat = pit_permutate(preds=yr_hat, perm=perms)

        # normalize the audios so that the maximum doesn't exceed 1
        if self.norm_if_exceed_1:
            max_vals = torch.max(torch.abs(yr_hat), dim=-1).values
            norm = torch.where(max_vals > 1, max_vals, 1)
            yr_hat = yr_hat / norm.unsqueeze(-1)

        return yr_hat

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        GS.on_predict_batch_end(self=self, outputs=outputs, batch=batch)

    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        return GS.configure_optimizers(
            self=self,
            optimizer=self.optimizer[0],
            optimizer_kwargs=self.optimizer[1],
            monitor='val/loss',
            lr_scheduler=self.lr_scheduler[0] if self.lr_scheduler is not None else None,
            lr_scheduler_kwargs=self.lr_scheduler[1] if self.lr_scheduler is not None else None,
        )

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     GS.on_load_checkpoint(self=self, checkpoint=checkpoint, ensemble_opts=self.ensemble, compile=self.compile)


class TrainCLI(BaseCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_neg_si_sdr_{val/neg_si_sdr:.4f}",
            "model_checkpoint.monitor": "val/neg_si_sdr",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,  # save all checkpoints
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        self.add_model_invariant_arguments_to_parser(parser)


if __name__ == '__main__':
    # python SharedTrainer.py --help
    cli = TrainCLI(
        TrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
