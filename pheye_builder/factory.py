from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from .phEYE import phEYE
from .wrapper_lm import phEYELMMixin
from .utils import extend_instance
from .encoder import Encoder


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    lang_decoder_path: str,
    tokenizer_path: str,
    dtype,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = True,
    cache_dir: Optional[str] = None,
    level: int = 2,
    encoder_dtype : torch.dtype = None,
    decoder_dtype : torch.dtype = None,
    use_dropout : bool = False,
    **pheye_kwargs,
):
    """
    Initialize a phEYE model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        phEYE: phEYE model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """

    vision_encoder = Encoder(clip_vision_encoder_path, level=level, dtype=encoder_dtype, use_dropout=use_dropout)


    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    #print(lang_decoder_path)
    lang_config = AutoConfig.from_pretrained(lang_decoder_path)
    #print(lang_config)
    lang_encoder = AutoModelForCausalLM.from_config(
        lang_config,
        #local_files_only=use_local_files,
        #trust_remote_code=True,
        torch_dtype=decoder_dtype
)

    lang_encoder.config.decoder_start_token_id = None
    lang_encoder.config.pad_token_id = text_tokenizer.pad_token_id

    # convert LM to phEYELM
    extend_instance(lang_encoder, phEYELMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)

    model = phEYE(
        vision_encoder,
        lang_encoder,
        vis_dim=vision_encoder.vision_model.config.hidden_size,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        dtype=dtype,
        **pheye_kwargs,
    )

    # Freeze all parameters
    model.lang_encoder.requires_grad_(False)
    assert sum(p.numel() for p in model.lang_encoder.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, cross_attn_layers, and LM input embeddings
    model.lang_encoder.cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)

    print(
        f"phEYE model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gpt": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "phi" : "model.layers"
}