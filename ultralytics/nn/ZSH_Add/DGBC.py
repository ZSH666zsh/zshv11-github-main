import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicBottConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicBottConv, self).__init__()
        mid_channels = in_channels // 2  # 适度压缩

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 动态卷积核
        self.conv3x3 = nn.Conv2d(mid_channels, mid_channels, 3,
                                 stride=stride, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(mid_channels, mid_channels, 5,
                                 stride=stride, padding=2, bias=False)

        # 动态权重
        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels // 8, 2, 1),  # 生成2个通道的权重
            nn.Softmax(dim=1)
        )

        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))

        w = self.weights(x)
        conv3 = self.conv3x3(x) * w[:, 0:1, :, :]  # w中提取的两个权重，形状为[batch_size, 1 , 1, 1]，将3x3卷积的输出乘以第一个权重
        conv5 = self.conv5x5(x) * w[:, 1:2, :, :]
        x = conv3 + conv5

        x = F.relu(self.bn2(self.conv2(x)))

        return F.relu(x + residual)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(MultiScaleFusion, self).__init__()
        out_channels = out_channels or in_channels

        # 多尺度
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )
        # 特征重组，输出到指定通道数
        self.reorg = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.reorg(out)
        return out


class DGBConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_type='GN'):
        super(DGBConv, self).__init__()
        self.out_channels = out_channels or in_channels

        # 动态路径
        self.path_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 2, 1),
            nn.Softmax(dim=1)
        )

        # 路径1：深度
        self.path1 = nn.Sequential(
            DynamicBottConv(in_channels, in_channels),
            DynamicBottConv(in_channels, self.out_channels),
            get_norm_layer(norm_type, self.out_channels, self.out_channels // 16),
            nn.ReLU()
        )

        # 路径2：多尺度
        self.path2 = nn.Sequential(
            MultiScaleFusion(in_channels, self.out_channels),
            get_norm_layer(norm_type, self.out_channels, self.out_channels // 16),
            nn.ReLU()
        )

        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, 1),
            get_norm_layer(norm_type, self.out_channels, self.out_channels // 16),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()
        if in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                get_norm_layer(norm_type, self.out_channels, self.out_channels // 16)
            )

    def forward(self, x):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x

        # 动态路径选择
        weights = self.path_selector(x)

        # 并行执行两条路径
        p1 = self.path1(x) * weights[:, 0:1, :, :]
        p2 = self.path2(x) * weights[:, 1:2, :, :]

        # 特征融合
        out = torch.cat([p1, p2], dim=1)
        out = self.enhance(out)

        return F.relu(out + residual)


def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)
