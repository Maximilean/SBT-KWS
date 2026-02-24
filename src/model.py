from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .subspectralnorm import SubSpectralNorm


class _DSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyDSCNN(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=(4, 1), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            _DSBlock(8,  16, stride=(2, 2)),
            _DSBlock(16, 24, stride=(2, 2)),
            _DSBlock(24, 32, stride=(1, 1)),
            _DSBlock(32, 48, stride=(1, 2)),
            _DSBlock(48, 48, stride=(1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, mel, T) → (B, 1, mel, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier[0](x)   # AdaptiveAvgPool2d → (B, 48, 1, 1)
        return self.classifier[1](x)  # Flatten → (B, 48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


class Conv1dNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        kernels: List[int],
        strides: List[int],
        channels: List[int],
        activation: torch.nn.Module,
        hidden_size: int,
    ):
        super().__init__()

        features = in_features

        module_list = []

        for kernel_size, stride, chs in zip(kernels, strides, channels):

            module_list.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=chs,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=chs,
                    ),
                    activation,
                    torch.nn.Conv1d(in_channels=chs, out_channels=chs, kernel_size=1),
                    torch.nn.BatchNorm1d(num_features=chs),
                    activation,
                    torch.nn.MaxPool1d(kernel_size=stride),
                ]
            )

            features = chs

        module_list.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(features, hidden_size),
                activation,
                torch.nn.Linear(hidden_size, n_classes),
                torch.nn.LogSoftmax(-1),
            ]
        )

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    def __init__(self, base_c, n_classes=12, n: Optional[List[int]] = None):
        super().__init__()
        self.num_classes = n_classes
        self.n = n if n is not None else [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, freq, time) -> (B, 1, freq, time)
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        return x.view(-1, x.shape[1])  # raw logits