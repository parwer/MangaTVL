"""LaMa generator (Fast Fourier Convolution ResNet), written for this project.

Architecture as described in "Resolution-robust Large Mask Inpainting with Fourier
Convolutions" (Suvorov et al., WACV 2022; reference implementation advimman/lama,
Apache-2.0). Module and parameter names follow that layout so the published Big-LaMa
style checkpoints (e.g. the manga fine-tune ``lama_large_512px``, MIT) load with
``strict=True``.

Big-LaMa generator, input = masked RGB + mask (4 channels):

    ReflectionPad(3) -> FFC 7x7 (4->64, local only)
    -> 3 x FFC 3x3 stride 2 (64->128->256->512; the last splits 25 % local / 75 % global)
    -> 18 x FFC residual blocks (local 128 + global 384 channels)
    -> concat -> 3 x [ConvTranspose 3x3 stride 2, BN, ReLU] (512->256->128->64)
    -> ReflectionPad(3) -> Conv 7x7 (64->3) -> sigmoid

The "global" branch of an FFC runs a 1x1 convolution in the frequency domain
(real FFT over H, W), which gives every layer an image-wide receptive field: that is
what lets LaMa continue screentone and long lines across the hole.
"""
from __future__ import annotations

import torch
from torch import nn


class FourierUnit(nn.Module):
    """1x1 conv + BN + ReLU on the real/imag parts of rfft2(x), then the inverse FFT."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_ch * 2, out_ch * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, _, h, w = x.shape
        f = torch.fft.rfftn(x.float(), dim=(-2, -1), norm="ortho")          # b, c, h, w//2+1 (complex)
        f = torch.stack((f.real, f.imag), dim=2)                             # b, c, 2, h, w'
        f = f.reshape(b, -1, *f.shape[3:])                                   # b, 2c (re/im interleaved), h, w'
        f = self.relu(self.bn(self.conv_layer(f.to(x.dtype))))
        f = f.float().reshape(b, -1, 2, *f.shape[2:])                        # b, c', 2, h, w'
        f = torch.complex(f[:, :, 0], f[:, :, 1])
        return torch.fft.irfftn(f, s=(h, w), dim=(-2, -1), norm="ortho").to(x.dtype)


class SpectralTransform(nn.Module):
    """Global-to-global path: 1x1 reduce -> FourierUnit -> residual -> 1x1 expand."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.fu = FourierUnit(mid, mid)
        self.conv2 = nn.Conv2d(mid, out_ch, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    """Fast Fourier Convolution: channels are split into a local part (spatial conv) and a
    global part (spectral). Four paths: l->l, l->g, g->l (spatial convs), g->g (spectral)."""

    def __init__(self, in_ch, out_ch, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0):
        super().__init__()
        in_g, out_g = int(in_ch * ratio_gin), int(out_ch * ratio_gout)
        in_l, out_l = in_ch - in_g, out_ch - out_g
        self.in_g, self.out_g = in_g, out_g

        def conv(i, o):
            if i == 0 or o == 0:
                return None
            return nn.Conv2d(i, o, kernel_size, stride, padding, bias=False, padding_mode="reflect")

        self.convl2l = conv(in_l, out_l)
        self.convl2g = conv(in_l, out_g)
        self.convg2l = conv(in_g, out_l)
        self.convg2g = SpectralTransform(in_g, out_g) if in_g and out_g else None

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_l = out_g = 0
        if self.convl2l is not None:
            out_l = self.convl2l(x_l)
        if self.convg2l is not None and x_g is not None:
            out_l = out_l + self.convg2l(x_g)
        if self.convl2g is not None:
            out_g = self.convl2g(x_l)
        if self.convg2g is not None and x_g is not None:
            out_g = out_g + self.convg2g(x_g)
        return out_l, out_g


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, ratio_gin=0.0, ratio_gout=0.0, stride=1, padding=0):
        super().__init__()
        self.ffc = FFC(in_ch, out_ch, kernel_size, ratio_gin, ratio_gout, stride, padding)
        out_g = int(out_ch * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_ch - out_g) if out_ch - out_g else nn.Identity()
        self.bn_g = nn.BatchNorm2d(out_g) if out_g else nn.Identity()
        self.act_l = nn.ReLU(inplace=True) if out_ch - out_g else nn.Identity()
        self.act_g = nn.ReLU(inplace=True) if out_g else nn.Identity()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l)) if torch.is_tensor(x_l) else x_l
        x_g = self.act_g(self.bn_g(x_g)) if torch.is_tensor(x_g) else x_g
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, ch, ratio=0.75):
        super().__init__()
        self.conv1 = FFC_BN_ACT(ch, ch, 3, ratio, ratio, padding=1)
        self.conv2 = FFC_BN_ACT(ch, ch, 3, ratio, ratio, padding=1)

    def forward(self, x):
        x_l, x_g = x
        y_l, y_g = self.conv2(self.conv1((x_l, x_g)))
        return x_l + y_l, x_g + y_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        return torch.cat([x_l, x_g], dim=1) if torch.is_tensor(x_g) else x_l


class LamaGenerator(nn.Module):
    """``forward(image, mask)``: image [B,3,H,W] in 0..1, mask [B,1,H,W] (1 = hole);
    H and W must be multiples of 8. Returns the inpainted image in 0..1."""

    def __init__(self, ngf: int = 64, n_downsampling: int = 3, n_blocks: int = 18, ratio: float = 0.75):
        super().__init__()
        layers: list[nn.Module] = [nn.ReflectionPad2d(3), FFC_BN_ACT(4, ngf, 7)]
        for i in range(n_downsampling):
            last = i == n_downsampling - 1
            layers.append(FFC_BN_ACT(ngf * 2 ** i, ngf * 2 ** (i + 1), 3, 0.0, ratio if last else 0.0,
                                     stride=2, padding=1))
        ch = ngf * 2 ** n_downsampling
        layers += [FFCResnetBlock(ch, ratio) for _ in range(n_blocks)]
        layers.append(ConcatTupleLayer())
        for i in range(n_downsampling):
            c = ngf * 2 ** (n_downsampling - i)
            layers += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, image, mask):
        return self.model(torch.cat([image * (1 - mask), mask], dim=1))


def load_lama(path, device="cpu") -> LamaGenerator:
    """Load a Big-LaMa checkpoint (raw state dict, or a training checkpoint with
    ``gen_state_dict``) into a LamaGenerator in eval mode."""
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = sd.get("gen_state_dict", sd) if isinstance(sd, dict) else sd
    gen = LamaGenerator()
    gen.load_state_dict(sd, strict=True)
    return gen.eval().to(device)
