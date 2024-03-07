import torch
import torch.nn as nn
import cfg


class Lss():
    def __init__(self):
        self.frustum = self.create_frustum().to('cuda')
        # (B, N, D, H, W, 3) = torch.Size([2, 6, 118, 32, 88, 3])
        self.geom = self.get_geometry(cfg.camera2lidar_rots,
                                      cfg.camera2lidar_trans,
                                      cfg.intrins,
                                      cfg.post_rots,
                                      cfg.post_trans)
        print(self.geom)

    def create_frustum(self):
        # 原图大小
        iH, iW = 128, 352
        # 特征图大小
        fH, fW = 8, 22
        # 原图的深度范围：4-45，每隔 1 米有个深度值
        dbound = [4, 45, 1]

        ds = (
            torch.arange(*dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        # 特征图的深度点到原图的映射。
        # (d, h, w, 3)=torch.Size([41, 8, 22, 3])
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
    ):
        B, N, _ = camera2lidar_trans.shape

        # 去掉图像增强
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # 将点转换到lidar坐标系
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # (B, N, D, H, W, 3) = torch.Size([1, 6, 118, 32, 88, 3])
        return points


def main():
    lss = Lss()


if __name__ == '__main__':
    main()
