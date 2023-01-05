import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x): # input:(B,C,D,H/4,W/4)=(4,32,192,128,160)，说实话不是很清楚为啥深度跟着变化
        # 先一路卷积在深度降维，再一路反卷积升维，过程中把每步卷积和反卷积对应的volume都累加起来传播
        conv0 = self.conv0(x) # (4,8,192,128,160)
        conv2 = self.conv2(self.conv1(conv0)) # (4,16,96,64,80)
        conv4 = self.conv4(self.conv3(conv2)) # (4,32,48,32,40)
        x = self.conv6(self.conv5(conv4)) # (4,64,24,16,20)
        x = conv4 + self.conv7(x) # （4，32，48，32，40） 反卷积升维
        x = conv2 + self.conv9(x) # (4,16,96,64,80)
        x = conv0 + self.conv11(x) # (4,8,192,128,160)
        x = self.prob(x) # (4,1,192,128,160)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet() # 特征提取网络
        self.cost_regularization = CostRegNet() #  cost volume正则化网络
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        """
        :param imgs:tuple元素大小3，每个元素（B,C,512，640）
        :param proj_matrices: tuple元素大小3，（B,4,4）
        :param depth_values: 在每个抽样深度的具体深度值，425-900+
        :return:
        """
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] #

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # 创建一个深度维度(B,C,D,H//4,W//4)
        volume_sum = ref_volume # 暂存
        volume_sq_sum = ref_volume ** 2 # 特征图的每个元素求平方
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs): # 遍历所有的源视角，将特征扭曲到cost volume中去
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values) # 构建volume
            if self.training:
                volume_sum = volume_sum + warped_volume # 把源视角的特征加到volume中去
                volume_sq_sum = volume_sq_sum + warped_volume ** 2 # 把平方项展开，这个是最前面的一项平方的和
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) # 这个方差计算公式比较奇怪

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1) # 直接softmax获得概率空间p
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # 概率图？
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1) # 概率空间每4个维度计算一个
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long() # 找到最大的那个下标
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1) # 得到概率图

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5 # 不是黑色的那些就算loss
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
