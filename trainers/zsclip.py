import torch
import torch.nn as nn
from tqdm import tqdm
from mm_dassl.engine import TRAINER_REGISTRY, TrainerX
from mm_dassl.optim import build_optimizer, build_lr_scheduler
import time

from clip import clip
from clip.model import convert_weights
from trainers.utils import * 
from trainers.utils import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT



BASE_TEMPLATES = {
    "OxfordPets": "a photo of a {}.",
    "OxfordFlowers": "a photo of a {}.",
    "FGVCAircraft": "a photo of a {}.",
    "DescribableTextures": "a photo of a {}.",
    "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101":"a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "SVHN": "a photo of a {}.",
    "Resisc45": "a photo of a {}.",
    "CLEVR": "a photo of a {}.",
    "LocMNIST": "a photo of a {}.",
    "ColourBiasedMNIST":"a photo of a {}.",
    'PCAM': "a photo of a {}.",
    "CIFAR10": "a photo of a {}.",
    "CIFAR100": "a photo of a {}.",
}

CONTEXT_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "SVHN": "This is a photo of a {}",
    "Resisc45": "This is a photo of a {}",
    "CLEVR": "This is a photo of {} objects",
    "LocMNIST": "This is a photo of {}",
    "ColourBiasedMNIST":"This is a photo of {}",
    "PCAM": 'this is a photo of {}',
    "CIFAR10": "This is a photo of {}",
    "CIFAR100": "This is a photo of {}",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a breed of pet.",

    "OxfordFlowers": "a vibrant photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",#, a type of aircraft.",
    "DescribableTextures": "a photo of fine patterns of a {}-textured surface",
    "EuroSAT": "a photo of centered satellite top-down view of {} location.",
    "StanfordCars": "a photo of a car model {}.",
    "Food101": "a close up photo showing delicious details of {}, a type of food.",
    # "SUN397": "a photo of a {} environment.",
        # "SUN397": "a photo showing good view of a {} scene.",
        "SUN397": "a photo showing good view of a {} location.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person actively engaged in {}.",
    # "ImageNet": "a photo of a {}.",
    "ImageNet": "a bad photo photo of {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "SVHN": "a pixelated street sign of the number {}",
    "Resisc45": "a satellite imagery of a {} location",
    "CLEVR": "a photo of {} colored objects",
    "LocMNIST": "a blurry photo of number {}",
    "ColourBiasedMNIST":"a pixelated number {} in colored backround",
    'PCAM': "a photo of tissue region {} ",
    "CIFAR10": "a pixelated photo of a {}",
    "CIFAR100": "a pixelated photo of a{}",
}

# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}.",

#     "OxfordFlowers": "a photo of a {}.",
#     "FGVCAircraft": "a photo of an aircraft {}.",#, a type of aircraft.",
#     "DescribableTextures": "a photo of a {}.",
#     "EuroSAT": "a photo of a {}.",
#     "StanfordCars": "a photo of a car model {}.",
#     "Food101": "a close-up photo showing delicious details of {}, a type of food.",
#     "SUN397": "a photo showing good view of a {} location.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a {}.",
#     # "ImageNet": "a photo of a {}.",
#     "ImageNet": "a bad photo photo of {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
#     "SVHN": "a photo of a {}",
#     "Resisc45": "a satellite imagery of a {} location",
#     "CLEVR": "a photo of {} colored objects",
#     "LocMNIST": "a blurry photo of number {}",
#     "ColourBiasedMNIST":"a pixelated number {} in colored backround",
#     'PCAM': "a photo of tissue region {} ",
#     "CIFAR10": "a blurry photo of {}",
#     "CIFAR100": "a blurry photo of {}",
# }

    
@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.float()

        clip_model.to(self.device)
        clip_model.eval()

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Text Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.to(self.device)
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits