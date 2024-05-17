import argparse
import os
import pickle

from diffusers import ConfigMixin, ModelMixin
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from datasets import load_dataset, load_from_disk, Image
from tqdm.auto import tqdm

# The `ImageEncoder` class is a model that takes image files as input, processes them, and encodes
# them using an encoder model.
class ImageEncoder(ModelMixin, ConfigMixin):
    def __init__(self, image_processor, encoder_model):
        """
        The function initializes an object with an image processor, an encoder model, and an image
        loader.
        
        :param image_processor: The image_processor parameter is an object or model that is responsible
        for processing or manipulating images. It could include tasks such as resizing, cropping,
        filtering, or any other image processing operations
        :param encoder_model: The `encoder_model` parameter is an object that represents a machine
        learning model used for encoding or feature extraction. It is likely used in the context of
        image processing or computer vision tasks. The specific details of the `encoder_model` would
        depend on the implementation and the specific library or framework being used
        """
        super().__init__()
        self.processor = image_processor
        self.encoder = encoder_model
        self.img_loader = Image(decode=True)
        
    def forward(self, x):
        """
        The forward function takes an input x, passes it through the encoder, and returns the output.
        
        :param x: The parameter "x" represents the input data that will be passed through the encoder
        :return: The output of the encoder.
        """
        x = self.encoder(x)
        return x
        
    @torch.no_grad()
    def encode(self, image_files):
        """
        The function takes a list of image files, encodes them using a pre-trained model, and returns
        the embeddings of the images.
        
        :param image_files: The `image_files` parameter is a list of file paths to the image files that
        you want to encode
        :return: the embeddings of the input images.
        """
        self.eval()
        images = [self.img_loader.decode_example(image_file) for image_file in image_files]
        x = self.processor(images, return_tensors="pt")['pixel_values']
        y = self(x)
        y = y.last_hidden_state
        embedings = y[:,0,:] 
        return embedings

def main(args):
    """
    The main function loads a dataset of images, encodes each image using a pre-trained ViT model, and
    saves the encoded representations to a pickle file.
    
    :param args: The `args` parameter is a dictionary or object that contains various arguments or
    options for the main function. It is used to pass values or configurations to the function
    """
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    image_encoder = ImageEncoder(processor, extractor)
    
    if args.dataset_name_or_path is not None:
        if os.path.exists(args.dataset_name_or_path):
            dataset = load_from_disk(args.dataset_name_or_path)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
            
    encodings = {}
    for image in tqdm(dataset.to_pandas()['image']):
        encodings[image['path']] = image_encoder.encode([image])
    pickle.dump(encodings, open(args.output_file, "wb"))
    
# The `if __name__ == "__main__":` block is a common Python idiom that allows a script to be executed
# as a standalone program or imported as a module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pickled image encodings for dataset of image files.")
    parser.add_argument("--dataset_name_or_path", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="encodings.p")
    parser.add_argument("--use_auth_token", type=bool, default=False)
    args = parser.parse_args()
    main(args)