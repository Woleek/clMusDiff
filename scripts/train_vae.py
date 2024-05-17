# based on https://github.com/CompVis/stable-diffusion/blob/main/main.py

import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from datasets import load_dataset, load_from_disk
from diffusers.pipelines.audio_diffusion import Mel
from ldm.util import instantiate_from_config
from librosa.util import normalize
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only 
from torch.utils.data import DataLoader, Dataset

from audiodiffusion.utils import convert_ldm_to_hf_vae


# The `AudioDiffusion` class is a dataset class that loads audio spectrogram data and converts it into
# a format suitable for training a model.
class AudioDiffusion(Dataset):
    def __init__(self, model_id, channels=3):
        """
        The function initializes an object with a model ID and an optional number of channels, and loads
        a dataset from disk or from a remote source.
        
        :param model_id: The `model_id` parameter is used to specify the identifier or path of the
        model. It can be either a local path or a pre-trained model identifier from the Hugging Face
        Model Hub
        :param channels: The `channels` parameter is used to specify the number of channels in the input
        data. In computer vision tasks, an image can have multiple channels, such as RGB images with 3
        channels (Red, Green, Blue). By default, the `channels` parameter is set to 3, indicating,
        defaults to 3 (optional)
        """
        super().__init__()
        self.channels = channels
        if os.path.exists(model_id):
            self.hf_dataset = load_from_disk(model_id)["train"]
        else:
            self.hf_dataset = load_dataset(model_id)["train"]

    def __len__(self):
        """
        The above function returns the length of the `hf_dataset` attribute.
        :return: The length of the `hf_dataset` attribute.
        """
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        The function takes an index as input and returns a dictionary containing a preprocessed image.
        
        :param idx: The `idx` parameter represents the index of the item you want to retrieve from the
        dataset. It is used to access a specific item in the dataset
        :return: The code is returning a dictionary with a key "spec" and the value being the processed
        image.
        """
        image = self.hf_dataset[idx]["spec"]
        if self.channels == 3:
            image = image.convert("RGB")
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width, self.channels))
        image = (image / 255) * 2 - 1
        return {"spec": image}


# The `AudioDiffusionDataModule` class is a PyTorch Lightning data module that provides a training
# dataloader for audio diffusion data.
class AudioDiffusionDataModule(pl.LightningDataModule):
    def __init__(self, model_id, batch_size, channels):
        """
        The function initializes an object with a given model ID, batch size, and number of channels,
        and sets the number of workers to 1.
        
        :param model_id: The model_id parameter is used to specify the ID of the audio diffusion model
        that will be used. This ID is typically used to load the corresponding pre-trained model or to
        identify the specific configuration of the model
        :param batch_size: The batch size is the number of samples that will be processed together in
        each iteration of the training or inference process. It determines how many samples will be fed
        into the model at once
        :param channels: The "channels" parameter represents the number of audio channels in the
        dataset. It determines the number of audio channels that will be used for training or inference
        with the model
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset = AudioDiffusion(model_id=model_id, channels=channels)
        self.num_workers = 1

    def train_dataloader(self):
        """
        The function returns a DataLoader object for training data with specified batch size and number
        of workers.
        :return: a DataLoader object.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# The `ImageLogger` class is a callback that logs audio data as images at a specified interval.
class ImageLogger(Callback):
    def __init__(self, every=1000, hop_length=512, sample_rate=22050, n_fft=2048):
        """
        The function initializes an object with specified parameters for audio processing.
        
        :param every: The "every" parameter determines the interval at which a certain action or
        calculation should be performed. It is measured in units of time or iterations, depending on the
        context, defaults to 1000 (optional)
        :param hop_length: The hop length is the number of samples between consecutive frames in the
        audio signal. It determines the time resolution of the analysis. A smaller hop length provides
        higher time resolution but requires more computational resources, defaults to 512 (optional)
        :param sample_rate: The sample rate is the number of samples of audio that are captured per
        second. It is typically measured in Hertz (Hz). In this case, the sample rate is set to 22050
        Hz, which means that 22050 samples of audio are captured per second, defaults to 22050
        (optional)
        :param n_fft: The number of samples in each frame of the Fast Fourier Transform (FFT). It
        determines the frequency resolution of the spectrogram, defaults to 2048 (optional)
        """
        super().__init__()
        self.every = every
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft

    @rank_zero_only
    def log_images_and_audios(self, pl_module, batch):
        """
        The function `log_images_and_audios` logs images and corresponding audios for training purposes.
        
        :param pl_module: pl_module is an instance of a PyTorch Lightning module. It is used to access
        the methods and attributes of the module within the function
        :param batch: The `batch` parameter is a batch of input data that is passed to the
        `log_images_and_audios` function. It is used to generate images and audios for logging purposes
        """
        pl_module.eval()
        with torch.no_grad():
            images = pl_module.log_images(batch, split="train")
        pl_module.train()

        image_shape = next(iter(images.values())).shape
        channels = image_shape[1]
        mel = Mel(
            x_res=image_shape[2],
            y_res=image_shape[3],
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        )

        for k in images:
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1.0, 1.0)
            images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = torchvision.utils.make_grid(images[k])

            tag = f"train/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

            images[k] = (images[k].numpy() * 255).round().astype("uint8").transpose(0, 2, 3, 1)
            for _, image in enumerate(images[k]):
                audio = mel.image_to_audio(
                    Image.fromarray(image, mode="RGB").convert("L")
                    if channels == 3
                    else Image.fromarray(image[:, :, 0])
                )
                pl_module.logger.experiment.add_audio(
                    tag + f"/{_}",
                    normalize(audio),
                    global_step=pl_module.global_step,
                    sample_rate=mel.get_sample_rate(),
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        The function `on_train_batch_end` is a callback function that is called at the end of each
        training batch, and it logs images and audios every `self.every` batches.
        
        :param trainer: The `trainer` parameter is an instance of the `pl.Trainer` class. It is
        responsible for managing the training process and provides various methods and properties for
        controlling the training loop
        :param pl_module: The `pl_module` parameter refers to the PyTorch Lightning module that is being
        trained. It is an instance of a class that inherits from `pl.LightningModule`
        :param outputs: The `outputs` parameter is a list containing the output of the forward pass of
        the model for the current batch. It typically contains the predicted values or logits for each
        input in the batch
        :param batch: The "batch" parameter represents the current batch of data being processed during
        training. It typically contains a batch of input samples and their corresponding labels
        :param batch_idx: The `batch_idx` parameter represents the index of the current batch being
        processed during training. It starts from 0 and increments by 1 for each batch
        :return: If the condition `(batch_idx + 1) % self.every != 0` is true, then the function
        `log_images_and_audios(pl_module, batch)` is called. If the condition is false, nothing is
        returned.
        """
        if (batch_idx + 1) % self.every != 0:
            return
        self.log_images_and_audios(pl_module, batch)


# The `HFModelCheckpoint` class is a subclass of `ModelCheckpoint` that saves checkpoints of a model
# trained using PyTorch Lightning and converts the saved checkpoints from a custom format to the
# Hugging Face format.
class HFModelCheckpoint(ModelCheckpoint):
    def __init__(self, ldm_config, hf_checkpoint, *args, **kwargs):
        """
        The function initializes an object with the given ldm_config, hf_checkpoint, and optional
        arguments.
        
        :param ldm_config: The `ldm_config` parameter is used to pass the configuration settings for the
        LDM (Language Data Model). It contains information such as the model architecture,
        hyperparameters, and other settings specific to the LDM
        :param hf_checkpoint: The `hf_checkpoint` parameter is used to specify the path or name of the
        Hugging Face model checkpoint. This checkpoint is a pre-trained model that can be used for
        various natural language processing tasks, such as text classification or language generation
        """
        super().__init__(*args, **kwargs)
        self.ldm_config = ldm_config
        self.hf_checkpoint = hf_checkpoint
        self.sample_size = None

    def on_train_batch_start(self, batch):
        """
        The function sets the sample size attribute to the shape of the "spec" key in the batch
        dictionary if it is not already set.
        
        :param batch: The `batch` parameter is a dictionary that contains the data for a single batch
        during training. It typically includes input features and target labels for the batch. In this
        case, it seems that the batch dictionary has a key called "spec" which represents the input
        features for the batch. The shape of
        """
        if self.sample_size is None:
            self.sample_size = list(batch["spec"].shape[1:3])

    def on_train_epoch_end(self, trainer, pl_module):
        """
        The function `on_train_epoch_end` updates the resolution of a model and converts it to a
        different format.
        
        :param trainer: The `trainer` parameter is an instance of the PyTorch Lightning `Trainer` class.
        It is responsible for managing the training process and provides various methods and properties
        related to training
        :param pl_module: pl_module is an instance of the PyTorch Lightning module that is being
        trained. It contains the model architecture, optimizer, and other training-related components
        """
        ldm_checkpoint = self._get_metric_interpolated_filepath_name({"epoch": trainer.current_epoch}, trainer)
        super().on_train_epoch_end(trainer, pl_module)
        self.ldm_config.model.params.ddconfig.resolution = self.sample_size
        convert_ldm_to_hf_vae(ldm_checkpoint, self.ldm_config, self.hf_checkpoint, self.sample_size)


# The above code is a Python script that trains a Variational Autoencoder (VAE) using the ldm (latent
# diffusion model) framework. It takes command-line arguments for various parameters such as dataset
# name, batch size, configuration file paths, checkpoint directories, etc.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using ldm.")
    parser.add_argument("-d", "--dataset_name", type=str, default="Woleek/Img2Spec")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-c", "--ldm_config_file", type=str, default="config/ldm_autoencoder_kl.yaml")
    parser.add_argument("--ldm_checkpoint_dir", type=str, default="models/ldm-autoencoder-kl")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="models/autoencoder-kl")
    parser.add_argument("-r", "--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("-g", "--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--save_images_batches", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    config = OmegaConf.load(args.ldm_config_file)
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    model.image_key = config.model.image_key
    data = AudioDiffusionDataModule(
        model_id=args.dataset_name,
        batch_size=args.batch_size,
        channels=config.model.params.ddconfig.in_channels,
    )
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config.accumulate_grad_batches = args.gradient_accumulation_steps
    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(
        trainer_opt,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        callbacks=[
            ImageLogger(
                every=args.save_images_batches,
                hop_length=args.hop_length,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
            ),
            HFModelCheckpoint(
                ldm_config=config,
                hf_checkpoint=args.hf_checkpoint_dir,
                dirpath=args.ldm_checkpoint_dir,
                filename="{epoch:06}",
                verbose=True,
                save_last=True,
            ),
        ],
    )
    trainer.fit(model, data)
