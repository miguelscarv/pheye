import torch
from peft import LoraConfig, get_peft_model
from torch import nn
import os


class phEYE(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        vis_dim: int,
        dtype: torch.dtype,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        reduce_factor = 1,
        from_layer = 0
    ):
        """
        Args:
            vision_encoder (nn.Module): module with OpenCLIP model
            lang_encoder (nn.Module): HF causal language model
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.vis_dim = vis_dim
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_pheye(
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
            reduce_factor=reduce_factor,
            from_layer=from_layer,
            dtype=dtype
        )
        self._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: list,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values = None,
        use_cache: bool = False,
        device="cpu",
        is_textcaps = False
    ):
        """
        Forward pass of phEYE.

        Args:
            vision_x (list): Vision input
                shape (B, C, H, W)
            lang_x (torch.Tensor): Language input ids
                shape (B, txt_seq)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_pheye
        ), "Wrapper layers are not initialized. Please call `initialized_pheye` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x, device=device, is_textcaps=is_textcaps)

        #print(f"Text features shape: {lang_x.shape}")
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: list,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        device = "cpu",
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (list): Vision input
                shape (B, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x, device=device, repeat=num_beams)

        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            num_beams=num_beams,
            **kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x(self, vision_x: list, device="cpu", repeat = 1, is_textcaps = False):
        """
        Compute vision features by passing images through vision encoder and conditioning language model.
        Args:
            vision_x (list): Vision input
                shape (B, C, H, W)
        """
        if is_textcaps:
            vision_x = vision_x[::5]
            repeat = 5

        vision_x = self.vision_encoder(vision_x, device=device)

        if repeat > 1:
            vision_x = vision_x.repeat_interleave(repeat, dim=0)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


    def cache_media(self, vision_x: list, device="cpu"):
        """
        Cache vision_x features from list of images for log-likelihood evaluation
        This is not meant to be used to cache things for generate().
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, F, C, H, W)
        """
        self._encode_vision_x(vision_x=vision_x, device=device)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False

    def save_model(self, _path):
        os.mkdir(_path)
        torch.save(self.vision_encoder.state_dict(), _path+"vision_encoder.pt")
        torch.save(self.lang_encoder.state_dict(), _path+"lang_encoder.pt")

    def add_lora_decoder(self):

        config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
                lora_dropout=0.05,
                bias="none"
            )

        self.lang_encoder.old_decoder_blocks = get_peft_model(self.lang_encoder.old_decoder_blocks, config)

    def merge_and_unload(self):
        self.lang_encoder.old_decoder_blocks = self.lang_encoder.old_decoder_blocks.merge_and_unload()

