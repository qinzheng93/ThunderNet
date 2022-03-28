import torch
import torch.nn as nn


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size=3, block_idx=0):
        super(ShuffleNetV2Block, self).__init__()

        pad = kernel_size // 2

        self.block_idx = block_idx
        stride = 2 if block_idx == 0 else 1
        branch_out = out_channel - in_channel

        branch = [
            # pw
            nn.Conv2d(in_channel, mid_channel, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channel, mid_channel, kernel_size, stride, padding=pad, groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel),
            # pw linear
            nn.Conv2d(mid_channel, branch_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
        ]
        self.branch = nn.Sequential(*branch)
        if block_idx == 0:
            branch_left = [
                # pw
                nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding=pad, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                # dw
                nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            ]
            self.branch_left = nn.Sequential(*branch_left)

    def forward(self, x):
        if self.block_idx == 0:
            return torch.cat((self.branch_left(x), self.branch(x)), 1)
        else:
            x1, x2 = self.channel_shuffle(x)
            return torch.cat((x1, self.branch(x2)), 1)

    def channel_shuffle(self, x):
        batch_size, num_channel, H, W = x.data.size()
        x = x.reshape(batch_size * num_channel // 2, 2, H * W)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, batch_size, num_channel // 2, H, W)
        return x[0], x[1]
