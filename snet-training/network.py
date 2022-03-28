import torch
import torch.nn as nn
from blocks import ShuffleNetV2Block
from tensorboardX import SummaryWriter


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='thunder'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        # self.stage_repeats_thunder = [3, 7, 3]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        elif model_size == 'thunder':
            self.stage_out_channels = [-1, 24, 132, 264, 528]
            # self.stage_repeats = self.stage_repeats_thunder
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_channel = [132, 264, 528]
        stage_repeat_num = [4, 8, 4]
        in_channel = 24
        self.stage = []
        for idx_stage in range(3):
            layer = []
            out_channel = stage_channel[idx_stage]
            for idx_repeat in range(stage_repeat_num[idx_stage]):
                if idx_repeat == 0:
                    layer.append(ShuffleNetV2Block(in_channel, out_channel, out_channel // 2, 5, idx_repeat))
                else:
                    layer.append(ShuffleNetV2Block(in_channel // 2, out_channel, out_channel // 2, 5, idx_repeat))
                in_channel = out_channel
            self.stage.append(nn.Sequential(*layer))
        self.stage = nn.Sequential(*self.stage)
        # conv_last 是否删除
        '''
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        '''
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.stage(x)
        '''
        x = self.stage[0](x)
        x = self.stage[1](x)
        x = self.stage[2](x)
        '''
        # x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = ShuffleNetV2()
    print(model)
    writer = SummaryWriter('/home/alpc111/append0/tensorboard/runs/exp')

    test_data = torch.rand(5, 3, 224, 224)
    # writer.add_graph(model, input_to_model=test_data, verbose=False)
    test_outputs = model(test_data)
    print(test_outputs.size())
