import os
import time
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from astronomaly.base.base_pipeline import PipelineStage

try:
    import torchvision
    from torchvision import transforms, models
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    err_string = """pytorch, torchvision, kornia and byol_pytorch must be
    installed to use this module."""
    raise ImportError(f"{err_string} \n {e}")  

class AstronomalyDataset(Dataset):
    def __init__(self, image_dataset, transform=None):
        """
        Creates a pytorch compatible dataset object from an 
        Astronomaly ImageDataset object
        which implements the get_sample method.
        """
        self.image_dataset = image_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset.index)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        idx = self.image_dataset.index[i]
        image = self.image_dataset.get_sample(idx)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        if len(image.shape) > 2 and image.shape[-1] <= 4:
            # cvread reads in images with channel last but torch needs
            # channel first
            image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, idx

class DINOv3_Features(PipelineStage):
    """DINOv3 features from Astronomaly image cutouts."""

    def __init__(
                self, 
                image_size=128,
                normalize=True,
                grayscale=True,
                batch_size=32,
                num_workers=16,
                **kwargs):
        """Pretrained DINOv3 model and image transforms for feature extraction.

        The image dataset and batching follow the supplied BYOL baseline.
        This stage uses a pretrained model rather than training one.

        Parameters
        ----------
        image_size: int
            Kept for pipeline compatibility. This class does not resize images.
        normalize: bool
            Whether to apply the pretrained model's channel normalisation.
        grayscale: bool
            Whether to repeat a greyscale image across three channels.
        batch_size: int
            Number of images processed in each inference batch.
        num_workers: int
            The number of workers to use for the DataLoader.
        """

        super().__init__(**kwargs)

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labels = []


        # Input tensors use CUDA when available and the CPU otherwise.

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        # The transforms prepare Astronomaly cutouts for the model.
        transform_list = [transforms.ToPILImage()]

        if grayscale:
            # Radio cutouts are greyscale, while the model expects three channels.
            transform_list += [
                transforms.Grayscale(num_output_channels=3)
                ]
        transform_list += [
            transforms.ToTensor()
            ]
        
        if normalize:
            # Normalisation uses the channel statistics of the pretrained model.
            transform_list += [transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transforms = transforms.Compose(transform_list)
        # The ViT-H+/16 architecture and weights come from local files.

        self.model = torch.hub.load(
        "/Users/kamoheloleriba/Downloads/dinov3-main",
        "dinov3_vith16plus",
        source="local",
        weights="/Users/kamoheloleriba/Downloads/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
        )

        # Evaluation mode switches off training-specific behaviour in the model.
        self.model.eval()

        
       

    def extract_features_batched(self, image_dataset):
        """One DINOv3 embedding per image in dataset order.

        Parameters
        ----------
        image_dataset : Astronomaly ImageDataset object
            The dataset to extract features for.

        Returns
        -------
        pandas.DataFrame
            Embeddings indexed by the original source identifiers.
        """

        extraction_dataset = AstronomalyDataset(
            image_dataset, self.transforms)
        
        data_loader = DataLoader(
            extraction_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)
        
        all_feats = []
        self.model.eval()

        t1 = time.perf_counter()

        with torch.no_grad():
            for i, image_batch in enumerate(data_loader):
                image_batch = image_batch[0]
                images = image_batch.to(self.device)

                # The pretrained model returns an embedding for each image.
                with torch.no_grad():
                     embedding = self.model(images)
                     

                # Each embedding is stored as a Numpy array for the output table.
                feats = embedding.detach().cpu().numpy()
                all_feats.append(feats)

                t2 = (time.perf_counter() - t1) / 60
                if i % 100 == 0:
                    print(f"Processed batch {i+1}/{len(data_loader)}")
                    print(f"Total time taken: {t2:.2f} minutes")
                    print()

        # The batch arrays form one feature matrix in dataset order.
        all_feats = np.concatenate(all_feats, axis=0)

        if len(self.labels) == 0:
            self.labels = [f'feat{i}' for i in range(len(all_feats[0]))]

        return pd.DataFrame(
            data=all_feats, index=image_dataset.index, columns=self.labels)

    def _execute_function(self, image):
        """A DINOv3 embedding for one image.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the extracted image embedding.
        """

        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        if len(image.shape) > 2 and image.shape[-1] <= 4:
            # cvread reads in images with channel last but torch needs
            # channel first
            image = image.permute(2, 0, 1)
        
        processed_image = self.transforms(image)
        # The leading dimension marks this single cutout as a batch of one.
        processed_image = torch.unsqueeze(processed_image, 0).to(self.device)

        # Gradients are unnecessary because the model is not trained here.
        with torch.no_grad():
            embedding = self.model(processed_image)
    
            feats = embedding.cpu().numpy().squeeze()

            if len(self.labels) == 0:
                self.labels = [f'feat{i}' for i in range(len(feats))]

            return feats

        
