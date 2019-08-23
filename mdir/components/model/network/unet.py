import torch
import torch.nn as nn
import torch.nn.functional as F


class OrigUNet(nn.Module):

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            return self.relu2(self.conv2(self.relu1(self.conv1(x))))

    class SkipConnBlock(nn.Module):
        def __init__(self, nested, channels, in_channels=None):
            super().__init__()
            in_channels = channels//2 if in_channels is None else in_channels
            self.downconv = OrigUNet.ConvBlock(in_channels, channels)
            self.pool = nn.MaxPool2d(2)
            self.nested = nested
            self.convT = nn.ConvTranspose2d(channels*2, channels, 2, stride=2)
            self.upconv = OrigUNet.ConvBlock(channels * 2, channels)

        def forward(self, x):
            x1 = self.downconv(x)
            x2 = self.convT(self.nested(self.pool(x1)))
            return self.upconv(torch.cat([x1, x2], dim=1))


    def __init__(self, in_channels, out_channels, nested_levels=4, min_channels=64):
        super().__init__()
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}
        innerblock = OrigUNet.ConvBlock(min_channels * 2**(nested_levels-1), min_channels * 2**nested_levels)
        for i in range(nested_levels-1, 0, -1):
            innerblock = OrigUNet.SkipConnBlock(innerblock, min_channels * 2**i)
        self.outerblock = OrigUNet.SkipConnBlock(innerblock, min_channels, in_channels=in_channels)
        self.outconv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        return self.outconv(self.outerblock(x))


class P2pUNet(nn.Module):

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}

    class SkipConnBlock(nn.Module):
        def __init__(self, nested, outer_channels, inter_channels, conv_opts, batchnorm_opts,
                     batchnorm=True, dropout=False):
            super().__init__()
            # Downsample
            modules = [nn.Conv2d(outer_channels, inter_channels, **conv_opts)]

            # Nested
            if nested:
                if batchnorm:
                    modules += [nn.BatchNorm2d(inter_channels, **batchnorm_opts)]
                modules += [nn.LeakyReLU(0.2), nested]
            else:
                modules += [nn.ReLU()]

            nested_channels = inter_channels*(2 if nested else 1)

            # Upsample
            modules += [nn.ConvTranspose2d(nested_channels, outer_channels, **conv_opts)]
            if batchnorm:
                modules += [nn.BatchNorm2d(outer_channels, **batchnorm_opts)]
            if dropout:
                modules += [nn.Dropout(p=dropout)]
            modules += [nn.ReLU()]

            self.nested = nn.Sequential(*modules)

        def forward(self, x):
            return torch.cat([x, self.nested(x)], dim=1)


    def __init__(self, in_channels, out_channels, dropout=0, conv_opts=None, batchnorm_opts=None,
                 batchnorm=True, nested_levels=7):
        super().__init__()
        conv_opts = {**P2pUNet.conv_opts, **(conv_opts or {})}
        batchnorm_opts = batchnorm_opts or {}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512), (512, 512)][:nested_levels]
        blocks += [(512, 512, True)] * (nested_levels - len(blocks))
        innerblock = None
        for block in reversed(blocks):
            chan_in, chan_out, use_dropout = block if len(block) == 3 else block + (False,)
            innerblock = P2pUNet.SkipConnBlock(innerblock, chan_in, chan_out, conv_opts=conv_opts,
                                               batchnorm_opts=batchnorm_opts, batchnorm=batchnorm,
                                               dropout=dropout*use_dropout)

        self.outerblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, **conv_opts),
            nn.LeakyReLU(0.2),
            innerblock,
            nn.ConvTranspose2d(128, out_channels, **{**conv_opts, "bias": True}),
            nn.Tanh()
        )

    def forward(self, x):
        return self.outerblock(x)


class ShallowP2pUNet(nn.Module):

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1}

    class SkipConnBlock(nn.Module):
        def __init__(self, nested, outer_channels, inter_channels, conv_opts):
            super().__init__()
            # Downsample
            modules = [
                nn.Conv2d(outer_channels, inter_channels, **conv_opts),
                nn.ReLU(),
                nn.Conv2d(inter_channels, inter_channels, 1),
                nn.ReLU(),
            ]

            # Nested
            if nested:
                modules.append(nested)
            nested_channels = inter_channels*(2 if nested else 1)

            # Upsample
            modules += [
                nn.ConvTranspose2d(nested_channels, outer_channels, **conv_opts),
                nn.ReLU(),
                nn.Conv2d(outer_channels, outer_channels, 1),
                nn.ReLU(),
            ]

            self.nested = nn.Sequential(*modules)

        def forward(self, x):
            return torch.cat([x, self.nested(x)], dim=1)


    def __init__(self, in_channels, out_channels, conv_opts=None, nested_levels=4):
        super().__init__()
        conv_opts = {**ShallowP2pUNet.conv_opts, **(conv_opts or {})}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512)][:nested_levels]
        blocks += [(512, 512)] * (nested_levels - len(blocks))
        innerblock = None
        for chan_in, chan_out in reversed(blocks):
            innerblock = ShallowP2pUNet.SkipConnBlock(innerblock, chan_in, chan_out, conv_opts=conv_opts)

        self.outerblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, **conv_opts),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            innerblock,
            nn.ConvTranspose2d(128, 64, **conv_opts),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        return self.outerblock(x)


class OutconvP2pUNet(nn.Module):

    SkipConnBlock = P2pUNet.SkipConnBlock

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1}

    def __init__(self, in_channels, out_channels, conv_opts=None, batchnorm_opts=None, nested_levels=7,
                 outconv_channels=32, outconv_kernel=3, dropout=False, batchnorm=False):
        super().__init__()
        assert outconv_kernel % 2 == 1
        conv_opts = {**self.conv_opts, **(conv_opts or {})}
        batchnorm_opts = batchnorm_opts or {}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512)][:nested_levels]
        blocks += [(512, 512)] * (nested_levels - len(blocks))
        innerblock = None
        for chan_in, chan_out in reversed(blocks):
            innerblock = self.SkipConnBlock(innerblock, chan_in, chan_out, conv_opts=conv_opts,
                                            batchnorm_opts=batchnorm_opts,
                                            batchnorm=batchnorm, dropout=dropout)

        self.outerblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, **conv_opts),
            nn.LeakyReLU(0.2),
            innerblock,
            nn.ConvTranspose2d(128, outconv_channels, **conv_opts),
            nn.ReLU(),
            nn.Conv2d(outconv_channels, out_channels, outconv_kernel, padding=outconv_kernel//2)
        )

    def forward(self, x):
        return self.outerblock(x)


class OutconvP2pUNetDynamicInterpolate(nn.Module):

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1}
    upconv_opts = {'kernel_size': 3, 'stride': 1, 'padding': 1}

    class SkipConnBlock(nn.Module):
        def __init__(self, nested, outer_channels, inter_channels, conv_opts, upconv_opts, upsample,
                     batchnorm_opts, batchnorm=True, dropout=False):
            super().__init__()
            self.upsample = upsample

            # Downsample
            modules = [nn.Conv2d(outer_channels, inter_channels, **conv_opts)]
            if batchnorm:
                modules += [nn.BatchNorm2d(inter_channels, **batchnorm_opts)]
            modules += [nn.LeakyReLU(0.2)]

            # Nested
            if nested:
                modules.append(nested)
            nested_channels = inter_channels*(2 if nested else 1)

            self.down = nn.Sequential(*modules)

            # Upsample
            modules = [nn.Conv2d(nested_channels, outer_channels, **upconv_opts)]
            if batchnorm:
                modules += [nn.BatchNorm2d(outer_channels, **batchnorm_opts)]
            if dropout:
                modules += [nn.Dropout(p=dropout)]
            modules += [nn.ReLU()]

            self.up = nn.Sequential(*modules)

        def forward(self, x):
            size = x.shape[-2:]
            y = self.up(F.interpolate(self.down(x), size=size, mode=self.upsample))
            return torch.cat([x, y], dim=1)

    def __init__(self, in_channels, out_channels, conv_opts=None, upconv_opts=None, nested_levels=7,
                 upsample="bilinear", outconv_channels=32, outconv_kernel=3, dropout=False, batchnorm=False):
        super().__init__()
        assert outconv_kernel % 2 == 1
        self.upsample = upsample
        conv_opts = {**OutconvP2pUNetDynamicInterpolate.conv_opts, **(conv_opts or {})}
        upconv_opts = {**OutconvP2pUNetDynamicInterpolate.upconv_opts, **(upconv_opts or {})}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512)][:nested_levels]
        blocks += [(512, 512)] * (nested_levels - len(blocks))
        innerblock = None
        for chan_in, chan_out in reversed(blocks):
            innerblock = OutconvP2pUNetDynamicInterpolate.SkipConnBlock(innerblock, chan_in, chan_out,
                                                                        conv_opts=conv_opts, upconv_opts=upconv_opts,
                                                                        upsample=upsample, batchnorm_opts={},
                                                                        batchnorm=batchnorm, dropout=dropout)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 64, **conv_opts),
            nn.LeakyReLU(0.2),
            innerblock
        )

        self.up = nn.Sequential(
            nn.Conv2d(128, outconv_channels, **upconv_opts),
            nn.ReLU(),
            nn.Conv2d(outconv_channels, out_channels, outconv_kernel, padding=outconv_kernel//2)
        )

    def forward(self, x):
        size = x.shape[-2:]
        return self.up(F.interpolate(self.down(x), size=size, mode=self.upsample))


class InconvP2pUNet(nn.Module):

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1}

    def __init__(self, in_channels, out_channels, conv_opts=None, nested_levels=7):
        super().__init__()
        conv_opts = {**OutconvP2pUNet.conv_opts, **(conv_opts or {})}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512)][:nested_levels]
        blocks += [(512, 512)] * (nested_levels - len(blocks))
        innerblock = None
        for chan_in, chan_out in reversed(blocks):
            innerblock = P2pUNet.SkipConnBlock(innerblock, chan_in, chan_out, conv_opts=conv_opts,
                                               batchnorm_opts={}, batchnorm=False, dropout=False)

        self.outerblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, **conv_opts),
            nn.LeakyReLU(0.2),
            innerblock,
            nn.ConvTranspose2d(128, out_channels, **conv_opts),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.outerblock(x)


class AlignedP2pUNet(nn.Module):

    conv_opts = {'kernel_size': 4, 'stride': 2, 'padding': 1}

    def __init__(self, in_channels, out_channels, conv_opts=None, nested_levels=7):
        super().__init__()
        conv_opts = {**AlignedP2pUNet.conv_opts, **(conv_opts or {})}
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}

        blocks = [(64, 128), (128, 256), (256, 512)][:nested_levels]
        blocks += [(512, 512)] * (nested_levels - len(blocks))
        innerblock = None
        for chan_in, chan_out in reversed(blocks):
            innerblock = P2pUNet.SkipConnBlock(innerblock, chan_in, chan_out, conv_opts=conv_opts,
                                               batchnorm_opts={}, batchnorm=False, dropout=False)

        self.outerblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            innerblock,
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.outerblock(x)


if __name__ == "__main__":
    net = P2pUNet(3, 3)
    print(net)
    print("\n".join("%s: %s" % (x, y.numel()) for x, y in net.named_parameters()))
    print(sum(p.numel() for p in net.parameters()))
    print(net(torch.randn(1,3,512,512)).shape)
    net = OrigUNet(3, 3)
    #print(net)
    print("\n".join("%s: %s" % (x, y.numel()) for x, y in net.named_parameters()))
