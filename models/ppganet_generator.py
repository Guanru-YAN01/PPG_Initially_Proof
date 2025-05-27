import torch
import torch.nn as nn

class UNetGenerator1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 base_filters=2, kernel_size=3, stride=2, padding=1):
        super(UNetGenerator1D, self).__init__()
        # Encoder layers
        self.enc1 = self._conv_block(in_channels, base_filters, kernel_size, stride, padding)
        self.enc2 = self._conv_block(base_filters, base_filters*2, kernel_size, stride, padding)
        self.enc3 = self._conv_block(base_filters*2, base_filters*4, kernel_size, stride, padding)
        self.enc4 = self._conv_block(base_filters*4, base_filters*8, kernel_size, stride, padding)
        self.enc5 = self._conv_block(base_filters*8, base_filters*16, kernel_size, stride, padding)
        self.enc6 = self._conv_block(base_filters*16, base_filters*32, kernel_size, stride, padding)
        self.enc7 = self._conv_block(base_filters*32, base_filters*64, kernel_size, stride, padding)
        self.enc8 = self._conv_block(base_filters*64, base_filters*128, kernel_size, stride, padding)
        self.enc9 = self._conv_block(base_filters*128, base_filters*256, kernel_size, stride, padding)

        # Decoder layers (ConvTranspose1d) with skip connections
        self.dec9 = self._deconv_block(base_filters*256, base_filters*128, kernel_size, stride, padding)
        self.dec8 = self._deconv_block(base_filters*256, base_filters*128, kernel_size, stride, padding)
        self.dec7 = self._deconv_block(base_filters*192, base_filters*64, kernel_size, stride, padding)
        self.dec6 = self._deconv_block(base_filters*96, base_filters*32, kernel_size, stride, padding)
        self.dec5 = self._deconv_block(base_filters*48, base_filters*16, kernel_size, stride, padding)
        self.dec4 = self._deconv_block(base_filters*24, base_filters*8, kernel_size, stride, padding)
        self.dec3 = self._deconv_block(base_filters*12, base_filters*4, kernel_size, stride, padding)
        self.dec2 = self._deconv_block(base_filters*6, base_filters*2, kernel_size, stride, padding)
        self.dec1 = self._deconv_block(base_filters*3, base_filters, kernel_size, stride, padding)

        # Final output conv (applies to d1 with channels=base_filters)
        self.final = nn.Sequential(
            nn.Conv1d(base_filters, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Tanh()
        )

    def _conv_block(self, in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _deconv_block(self, in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        e9 = self.enc9(e8)

        # Decoder with skip connections
        d9 = self.dec9(e9)
        d9 = torch.cat([d9, e8], dim=1)
        d8 = self.dec8(d9)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.dec7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)

        # Final conv
        out = self.final(d1)
        return out