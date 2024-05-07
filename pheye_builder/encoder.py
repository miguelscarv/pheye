
from transformers import CLIPModel
from torch import nn
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import PIL
from PIL.Image import BICUBIC
import math
from torchvision import transforms
import torch.nn.functional as F


# level 4 which has 21 patches was being used in previous experiments so now I can't remove it or won't be able to load older models....
LEVELS_TO_PATCHES = {
    1 : 1,
    2 : 5,
    3 : 10,
    4 : 21
}

def cut_image_patches(image: PIL.Image, encoder_resolution: int = 224):

    coordinates = []

    width, height = image.size

    width_tiles = [i*encoder_resolution for i in range(math.ceil(width/encoder_resolution)-1)]
    width_tiles.append(width-encoder_resolution)
    height_tiles = [i*encoder_resolution for i in range(math.ceil(height/encoder_resolution)-1)]
    height_tiles.append(height-encoder_resolution)
    
    for w in width_tiles:
        for h in height_tiles:
            coordinates.append((w,h,w+encoder_resolution,h+encoder_resolution))

    cropped_images = [image.crop(c) for c in coordinates]

    return cropped_images

class Encoder(nn.Module):

    def __init__(self, clip_name, level = 2, dtype = None, use_dropout = True) -> None:
        super().__init__()

        if level not in LEVELS_TO_PATCHES:
            raise ValueError("Resolution not supported")
        
        self.n_patches = LEVELS_TO_PATCHES[level]
        self.vision_model = CLIPModel.from_pretrained(clip_name, torch_dtype=dtype).vision_model
        self.has_first_adapter = False
        self.image_size = self.vision_model.config.image_size
        self.patch_size = self.vision_model.config.patch_size
        self.use_dropout = use_dropout
        self.dtype = dtype

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.norm_lvl_1 = nn.LayerNorm(self.vision_model.config.hidden_size, dtype=dtype)
        self.norm_lvl_2 = nn.LayerNorm(self.vision_model.config.hidden_size, dtype=dtype)

        # this was being used in previous experiments so now I can't remove it or won't be able to load older models....
        self.norm_lvl_3 = nn.LayerNorm(self.vision_model.config.hidden_size, dtype=dtype)

        if level == 1:
            self.connector = nn.LayerNorm(self.vision_model.config.hidden_size, dtype=dtype)
        else:
            self.connector = Position(self.n_patches, self.vision_model.config.hidden_size, dtype=dtype)

            config_level2 = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "patch_embedding", "fc1", "fc2"],
                lora_dropout=0.05 if self.use_dropout else 0,
                bias="none"
            )
            self.vision_model = get_peft_model(self.vision_model, config_level2, "second")

    def add_first_level_adapter(self):

        config_224 = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "patch_embedding", "fc1", "fc2"],
            lora_dropout=0.05 if self.use_dropout else 0,
            bias="none"
        )

        self.vision_model.add_adapter("first", config_224)
        self.has_first_adapter = True


    def forward(self, images: list, device = "cpu", **kwargs):
        """
        shape (B, C, H, W) in list form
        """
        B = len(images)
        h = int((self.image_size/self.patch_size) ** 2 + 1)
        resized_images = {1: [], 2: []}

        for i in images:
            resized_images[1].append(self.image_transform(i.resize((self.image_size,self.image_size), resample=BICUBIC)))

            if self.n_patches == 5:
                for crop in cut_image_patches(i.resize((self.image_size * 2,self.image_size * 2), resample=BICUBIC), encoder_resolution=self.image_size):
                    resized_images[2].append(self.image_transform(crop))
            elif self.n_patches == 10:
                for crop in cut_image_patches(i.resize((self.image_size * 3,self.image_size * 3), resample=BICUBIC), encoder_resolution=self.image_size):
                    resized_images[2].append(self.image_transform(crop))


        vision_features = []
        for res, imgs in resized_images.items():
            if imgs != []:
                resized_images[res] = torch.stack(imgs, dim = 0).to(device)

                if res == 1 and self.has_first_adapter:
                    self.vision_model.set_adapter("first")
                    vision_features.append(self.norm_lvl_1(self.vision_model(resized_images[res]).last_hidden_state))
                elif res == 1:
                    with self.vision_model.disable_adapter():
                        vision_features.append(self.norm_lvl_1(self.vision_model(resized_images[res]).last_hidden_state))
                elif res == 2:
                    self.vision_model.set_adapter("second")
                    if self.n_patches == 5:
                        vision_features.append(self.norm_lvl_2(self.vision_model(resized_images[res]).last_hidden_state.view(B, h * 4, -1)))
                    elif self.n_patches == 10:
                        vision_features.append(self.norm_lvl_2(self.vision_model(resized_images[res]).last_hidden_state.view(B, h * 9, -1)))

        vision_features = torch.cat(vision_features, dim = 1)
        vision_features = self.connector(vision_features)

        return vision_features
        

class Position(nn.Module):

    def __init__(self, n_patches, dim, dtype) -> None:
        super().__init__()

        self.embedding = nn.Embedding(max(LEVELS_TO_PATCHES.values()), dim, dtype=dtype)
        self.n_patches = n_patches

        self.apply(self._init_weights)

    def forward(self, vision_features):

        batch_size, seq_len, dim = vision_features.size()
        single_encoder_dim = seq_len // self.n_patches
        device = vision_features.get_device()
        
        pos = torch.LongTensor(list(range(self.n_patches))).to(device if device != -1 else "cpu")
        pos = torch.repeat_interleave(self.embedding(pos).unsqueeze(0), single_encoder_dim, 1).expand(batch_size, -1, -1)
    
        return vision_features + pos


    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "fc1.weight" or name == "fc2.weight" or name == "to_out.weight":
                p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * self.n_decoder_layers)))