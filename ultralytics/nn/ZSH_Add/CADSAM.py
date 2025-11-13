import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3,
                                   padding=dilation,
                                   dilation=dilation,
                                   groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)  # 逐点卷积

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ASPP(nn.Module):
    def __init__(self, in_c, boolean):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 256, 1, 1, padding=0, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            # nn.Conv2d(in_c, 256, 3, 1, padding=3, dilation=3),
            DepthwiseSeparableConv(in_c, 256, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            # nn.Conv2d(in_c, 256, 3, 1, padding=5, dilation=5),
            DepthwiseSeparableConv(in_c, 256, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            # nn.Conv2d(in_c, 256, 3, 1, padding=7, dilation=7),
            DepthwiseSeparableConv(in_c, 256, dilation=7),  # 3x3的膨胀7卷积，最大感受野
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bool = boolean

    # 使用不同膨胀率的空洞卷积捕获多尺度信息，四个分支并行计算后沿通道拼接，保持空间分辨率不变。
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # if self.bool:
        #     print("1. ASPP中 前景 4通道融合的x：",x.size())
        # else:
        #     print("1. ASPP中 背景 4通道融合的x：",x.size())
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_c):
        super(CrossAttentionBlock, self).__init__()

        self.query_conv = nn.Conv2d(in_c, in_c // 8, 1)  # 生成紧凑1/8的查询向量，用于计算与其他特征的相似性
        self.key_conv = nn.Conv2d(in_c, in_c // 8, 1)  # 生成紧凑1/8的键向量，与查询向量交互
        self.value_conv = nn.Conv2d(in_c, in_c, 1)  # 保留原始通道维度，确保加权后的特征与输入尺度一致
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2, boolean):
        B, C, H, W = feat1.size()

        # view()用于改变张量形状，但不改变数据本身。如.view(B, -1, H * W)用来将原四维张量（B, C, H, W）转为三维，其中第二维自动计算
        Q = self.query_conv(feat1).view(B, -1, H * W)  # (B, C//8, N)
        K = self.key_conv(feat2).view(B, -1, H * W)  # (B, C//8, N)
        V = self.value_conv(feat2).view(B, -1, H * W)  # (B, C, N)

        attention = torch.bmm(Q.permute(0, 2, 1), K)
        attention = self.softmax(attention / (K.size(1) ** 0.5))

        out = torch.bmm(V, attention.permute(0, 2, 1)).view(B, C, H, W)

        # if boolean:
        #     print("2. CAB中 前景指导 返回的乘过注意力权重的out：",out.size())
        # else:
        #     print("2. CAB中 背景指导 返回的乘过注意力权重的out：",out.size())
        return out

class Pred_Layer(nn.Module):
    def __init__(self, in_c, width_multiple):
        super(Pred_Layer, self).__init__()
        self.out_channels = int(256 * width_multiple * 2)
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Conv2d(self.out_channels, 1, kernel_size=1, stride=1, padding=0)

    # enlayer：3x3卷积+BatchNorm+ReLU，用于特征增强，outlayer：1x1卷积将通道压缩为1，生成预测图
    def forward(self, fusion):
        x = self.enlayer(fusion)
        x1 = self.outlayer(x)
        # print(f"3. 双流融合后的fusion的C：{fusion.size()}")
        # print(f"4. 送入Pred_Layer输出enlayer增强后的特征enhanced_feat：{x.size()}，new_pred：{x1.size()}")
        # print("————————————————————————————————————————————————————————————————————————")
        return x, x1

class DSAM_CrossAttention(nn.Module):
    def __init__(self, width_multiple):
        super(DSAM_CrossAttention, self).__init__()
        self.ff_conv = None
        self.bf_conv = None
        self.cross_attention = None
        self.rgbd_pred_layer = None
        self.pred_conv = None
        self.width_multiple = width_multiple

        self.channel_adapter = None  # 新增通道转换层

    def forward(self, feat):
        if self.ff_conv is None:
            # 先动态转换通道数
            in_c = int(feat.size(1) * self.width_multiple * 2)
            self.channel_adapter = nn.Conv2d(feat.size(1), in_c, kernel_size=1).to(feat.device)
            # print("width_multiple：", self.width_multiple, "，输入通道数：", feat.size(1), "，调整后通道数：", in_c)

            self.ff_conv = ASPP(in_c,True)
            self.bf_conv = ASPP(in_c,False)
            self.cross_attention = CrossAttentionBlock(256 * 4)
            self.rgbd_pred_layer = Pred_Layer(256 * 4 * 2, self.width_multiple)
            self.pred_conv = nn.Conv2d(in_c, 1, kernel_size=(1, 1)).to(feat.device)  # 新增内部预测层，生成初始预测图

        # 尺寸获取方式：1.直接使用 feat.shape[2:]  2.显式解包 [_, _, H, W] = feat.size()，明确展示维度信息，便于后续可能使用H/W
        adapted_feat = self.channel_adapter(feat)
        _, _, H, W = adapted_feat.size()
        pred = F.interpolate(torch.sigmoid(self.pred_conv(adapted_feat)), size=(H, W), mode='bilinear', align_corners=True)

        # print(f"\n0. 输入调整后的adapted_feat：{adapted_feat.size()}，内部初始预测pred：{pred.size()}")

        ff_feat = self.ff_conv(adapted_feat * pred)
        bf_feat = self.bf_conv(adapted_feat * (1 - pred))

        ff_enhanced = self.cross_attention(ff_feat, bf_feat, True)
        bf_enhanced = self.cross_attention(bf_feat, ff_feat, False)

        # 将双流特征拼接后送入Pred_Layer
        fusion=torch.cat((ff_enhanced, bf_enhanced), dim=1)
        enhanced_feat, new_pred = self.rgbd_pred_layer(fusion)

        return enhanced_feat  # 仅返回增强后的特征图
