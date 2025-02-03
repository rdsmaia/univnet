import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .lvcnet import LVCBlock

MAX_WAV_VALUE = 32768.0


class FiLM(nn.Module):
    def __init__(self, hp):
        super(FiLM, self).__init__()
        self.gamma_nn = nn.Sequential(
            nn.Linear(
               in_features=hp.audio.speaker_cond_dim,
               out_features=hp.audio.n_mel_channels
               ),
        )
        self.beta_nn = nn.Sequential(
            nn.Linear(
               in_features=hp.audio.speaker_cond_dim,
               out_features=hp.audio.n_mel_channels
               ),
        )

    def forward(self, s):
        gamma = self.gamma_nn(s)
        beta = self.beta_nn(s)

        return gamma, beta


class Upsampler(nn.Module):
    def __init__(self, hp):
        super(Upsampler, self).__init__()
        self.in_channels = hp.audio.latents_dim
        self.mel_channel = hp.audio.n_mel_channels
        self.num_upsamples = 1

        stride = 2
        in_channels = self.in_channels
        self.ups = nn.ModuleList()
        for i in range(self.num_upsamples):
            self.ups.append(
                nn.utils.weight_norm(nn.ConvTranspose1d(
                in_channels//(2**i),
                in_channels//(2**(i+1)),
                2*stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2)))
            self.ups.append(nn.LeakyReLU(hp.gen.lReLU_slope))
            self.ups.append(nn.BatchNorm1d(in_channels//(2**(i+1))))
        self.post = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels//(2**(i+1)), 128, 7, 1, padding=3)),
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.BatchNorm1d(128),
            nn.utils.weight_norm(nn.Conv1d(128, self.mel_channel, 7, 1, padding=3)),
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.BatchNorm1d(self.mel_channel))

    def forward(self, x):
        for up_step in self.ups:
            x = up_step(x)
        x = self.post(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for i, l in enumerate(self.ups):
             if i % 3 == 0:
                nn.utils.remove_weight_norm(l)
        for i, l in enumerate(self.post):
            if i % 3 == 0:
                nn.utils.remove_weight_norm(l)


class EmbeddingLayer(nn.Module):
    def __init__(self, hp):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(hp.audio.codes_vocab_size, hp.audio.latents_dim)
        if hp.audio.codes_hop_length != hp.audio.latents_hop_length:
           self.scale_factor = hp.audio.codes_hop_length / hp.audio.latents_hop_length
        else:
           self.scale_factor = None

    def forward(self, c):
        x = self.embedding(c).permute(0,2,1)
        if self.scale_factor is not None:
            return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = hp.audio.n_mel_channels
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size
        self.latents_dim = hp.audio.latents_dim

        # hop length between mel spectrograms and audio
        self.mel_ar_token_ratio = hp.audio.latents_hop_length // hp.audio.hop_length

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in hp.gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.audio.n_mel_channels,
                    stride=stride,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(hp.gen.noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

        self.embedding_layer = EmbeddingLayer(hp)
        self.upsampler = Upsampler(hp)
        self.ar_tokens_to_mel_spec_ratio = hp.audio.latents_hop_length // hp.audio.hop_length - 1
        self.noise_dim = hp.gen.noise_dim

        # conditioning layer
        self.cond_layer = FiLM(hp) if hp.audio.use_speaker_cond else None

    def forward(self, c, s=None):
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        '''

        # embedding layer
        c_emb = self.embedding_layer(c)

        # upsampler
        c_emb = self.upsampler(c_emb)

        # conditioning layer
        if self.cond_layer:
            gamma, beta = self.cond_layer(s)
            c_emb = c_emb * gamma.unsqueeze(2) + beta.unsqueeze(2)
            c_emb = c_emb

        # noise
        z = torch.randn(c.shape[0],
                        self.noise_dim,
                        c_emb.shape[2] * self.ar_tokens_to_mel_spec_ratio).to(c_emb.device)

        # pre-processing
        z = self.conv_pre(z)                # (B, c_g, L)

        # noise processing with conditioning
        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c_emb)             # (B, c_g, L * s_0 * ... * s_i)

        # post-processing
        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, s=None):

        audio = self.forward(c, s)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio

if __name__ == '__main__':
    hp = OmegaConf.load('../config/default.yaml')
    model = Generator(hp)

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
