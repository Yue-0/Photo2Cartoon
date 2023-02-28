import torch

nn = torch.nn
Layer = nn.Module
Sequential = nn.Sequential

ReLU = nn.ReLU
TanH = nn.Tanh
Sigmoid = nn.Sigmoid
LeakyReLU = nn.LeakyReLU

Conv2D = nn.Conv2d
TransposedConv2D = nn.ConvTranspose2d


class Concat:
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, *inputs):
        return torch.cat(inputs, self.axis)


class ConvBlock(Layer):
    def __init__(self,
                 conv,
                 in_channels,
                 out_channels,
                 activation_function,
                 activation_first,
                 bn,
                 dropout,
                 *act_params,
                 **conv_params):
        super(ConvBlock, self).__init__()
        block = [conv(in_channels, out_channels, **conv_params)]
        if bn:
            block.append(nn.BatchNorm2d(out_channels))
        if dropout:
            block.append(nn.Dropout())
        if activation_first:
            block.insert(0, activation_function(*act_params))
        else:
            block.append(activation_function(*act_params))
        self.block = Sequential(*block)

    def forward(self, *inputs):
        output = self.block(inputs[0])
        return torch.cat([output, inputs[1]], 1) if len(inputs) > 1 else output
