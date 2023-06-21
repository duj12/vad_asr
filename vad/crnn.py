import os
from typing import Optional, Tuple

import torch
from torch import nn


def crnn(inputdim, left_samples: int, right_samples: int,
         down_sample_factor: int, mel_hop_length: int, device: str, **kwargs):
    model = CRNN(inputdim, 2, left_samples, right_samples, down_sample_factor,
                 mel_hop_length, **kwargs)
    curfile_dir = os.path.dirname(__file__)
    state = torch.load(os.path.join(curfile_dir, "stream-hard.pt"),
                       map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model


class Block2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class MeanPool(nn.Module):

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""

    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(self.transform(logits))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CRNN(nn.Module):

    def __init__(self, inputdim, outputdim, left_samples: int,
                 right_samples: int, down_sample_factor: int,
                 mel_hop_length: int, **kwargs):
        super().__init__()
        self.left_samples = left_samples
        self.right_samples = right_samples
        self.down_sample_factor = down_sample_factor
        self.mel_hop_length = mel_hop_length

        self.gru_vector = None

        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=kwargs.get('gru_bidirection', True),
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(
            kwargs.get('temppool', 'linear'),
            inputdim=256 if kwargs.get('gru_bidirection', True) else 128,
            outputdim=outputdim)
        self.outputlayer = nn.Linear(
            256 if kwargs.get('gru_bidirection', True) else 128, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:

        cnn_feature = self.forward_stream_vad_cnn(feature)
        cnn_feature_to_gru = cnn_feature[:, self.left_samples //
                                         self.down_sample_factor:
                                         -self.right_samples //
                                         self.down_sample_factor, :]
        gru_output, self.gru_vector = self.forward_stream_vad_gru(
            cnn_feature_to_gru, self.gru_vector)

        _, time, _ = feature.shape
        output_frame_num = time - (
            (self.left_samples + self.right_samples) // self.mel_hop_length)
        return self.forward_stream_vad_upsample(gru_output, output_frame_num)

    def forward_stream_vad_cnn(self, x: torch.Tensor):
        """
            x:torch.Size([1, 9, 128]), torch.float32, [1, 60, 64], [1, 47, 64]
            [1, 5, 128]
        """
        x = x.unsqueeze(1)
        x = self.features(x)  # [1, 1, 60, 64], [1, 1, 47, 64]
        x = x.transpose(1, 2).contiguous().flatten(-2)
        return x

    def forward_stream_vad_gru(
            self, x: torch.Tensor,
            h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

            x: torch.Size([1, 9, 128]), torch.float32
            h: torch.Size([1, 1, 128]), torch.float32
        """
        x, h_ = self.gru(x, h)
        vad_post = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        return vad_post, h_  # torch.Tensor,float32,[1, 9, 2]; [1,1,128]

    def forward_stream_vad_upsample(
            self, vad_post: torch.Tensor,
            upsample_time_dimension: int) -> torch.Tensor:
        """
        参数：
            vad_post: torch.Size([1, 9, 2]),torch.float32
            upsample_time_dimension: torch.Tensor, [], int64
        返回：
            vad_post：一个三维的 torch.Tensor，数据类型为 torch.float32。第一维长度固
                定为 1，第二维长度是变化的，第三维长度固定为 2。vad_post[0][i][0] 表示
                静音概率，vad_post[0][i][1] 表示语音概率。
        """
        vad_post = torch.nn.functional.interpolate(
            vad_post.transpose(1, 2),
            upsample_time_dimension,
            mode='linear',
            align_corners=False).transpose(1, 2)
        return vad_post  # torch.Tensor,[1, 36, 2]
