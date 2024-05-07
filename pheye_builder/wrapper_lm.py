import torch.nn as nn
from .xattn import CrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


class WrapperLayer(nn.Module):
    """
    WrapperLayer is a wrapper around the CrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, cross_attn_layer, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.cross_attn_layer = cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        if self.cross_attn_layer is not None:
            self.cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            lang_x = self.cross_attn_layer(
                lang_x,
                self.vis_x
            )
            
        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )

        return lang_x


class phEYELMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_pheye(
        self,
        lang_hidden_size,
        vis_hidden_size,
        dtype,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        reduce_factor=1,
        from_layer=0
    ):
        """
        Initialize phEYE by adding a new cross attn to the decoder.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim_text=lang_hidden_size, dim_visual=vis_hidden_size, reduce_factor=reduce_factor, layer_idx=layer_idx, n_decoder_layers=len(self.old_decoder_blocks), dtype=dtype
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0 and layer_idx >= from_layer
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_pheye_layers(gradient_checkpointing)
        self.initialized_pheye = True
        self._use_cached_vision_x = False

    def init_pheye_layers(self, gradient_checkpointing):
        """
        Re initializes the WrapperLayers.
        Propagates any changes made to self.cross_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    WrapperLayer(
                        cross_attn_layer, decoder_layer, gradient_checkpointing
                    )
                    for cross_attn_layer, decoder_layer in zip(
                        self.cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        if not self.initialized_pheye:
            raise ValueError(
                "phEYE layers are not initialized. Please call `init_pheye` first."
            )

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)