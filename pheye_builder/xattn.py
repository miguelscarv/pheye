"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn
import math

def exists(val):
    return val is not None


class FeedForward(nn.Module):

    def __init__(self, dim, dtype, reduce_factor = 1):
        super().__init__()
        mult = 4
        self.norm = nn.LayerNorm(dim, dtype=dtype)
        inner_dim = int(dim * mult) // reduce_factor

        self.fc1 = nn.Linear(dim, inner_dim, dtype=dtype)
        self.fc2 = nn.Linear(inner_dim, dim, dtype=dtype)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

# cross attention
class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_visual,
        dtype,
        dim_head=64,
        reduce_factor=1
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        max_dim = max(dim_text, dim_visual)
        self.heads = max_dim // dim_head
        assert max_dim % dim_head == 0, f"Number of heads in CrossAttention is not an int - {self.heads}"
        inner_dim = max_dim // reduce_factor

        self.norm = nn.LayerNorm(dim_text, dtype=dtype)

        self.to_q = nn.Linear(dim_text, inner_dim, dtype=dtype)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, dtype=dtype)
        #self.to_kv_second = nn.Linear(dim_visual, inner_dim * 2)
        self.to_out = nn.Linear(inner_dim, dim_text, dtype=dtype)
        #self.g = []
        #self.l = []

    def forward(self, x, media):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, txt_seq, D_txt)
            media (torch.Tensor): image features
                shape (B, img_seq, D_img) where img_seq is the number of concatenated features from the ViT. For example:
                for an encoder of 224x224 with patch size 14 and processing images of 896x896 (with 3 levels) it will be (1 + 4 + 16) * 257 = 5397
        """
        
        h = self.heads

        x = self.norm(x)
        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        """k_s, v_s = self.to_kv(media[:, 257:, :]).chunk(2, dim=-1)
        k = torch.cat((k, k_s), 1)
        v = torch.cat((v, v_s), 1)"""
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        attn = sim.softmax(dim=-1)
        #idk = torch.mean(attn.squeeze()[:, 65:, :], (0, 1))
        #self.g.append(torch.sum(idk[:257]).item())
        #self.l.append(torch.sum(idk[257:]).item())
        
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

# cross attention
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_visual,
        dtype,
        dim_head=64,
        reduce_factor = 1,
        layer_idx=0,
        n_decoder_layers = 24
    ):
        super().__init__()
        self.attn = CrossAttention(
            dim_text=dim_text,
            dim_visual=dim_visual,
            dim_head=dim_head,
            reduce_factor=reduce_factor,
            dtype=dtype
        )

        self.ff = FeedForward(dim_text, reduce_factor=reduce_factor, dtype=dtype)
        self.layer_idx = layer_idx
        self.n_decoder_layers = n_decoder_layers

        self.apply(self._init_weights)

    def forward(
        self,
        x,
        media
    ):

        x = (
            self.attn(
                x,
                media
            )
            + x
        )
    

        x = self.ff(x) + x
        
        return x

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
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
            if name == "fc2.weight" or name == "to_out.weight":
                p.data.normal_(mean=0.0, std=(0.01 / math.sqrt(2 * max(self.n_decoder_layers, 36))))