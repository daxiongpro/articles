import torch
import torch.nn as nn


def create_frustum():
    # 原图大小
    iH, iW = 128, 352
    # 特征图大小
    fH, fW = 8, 22
    # 深度范围：4-45，每隔 1 米有个深度值
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
    # (d, h, w, 3) 特征图上位置在 (h, w) 处，深度为 d 的点在原图上的位置
    frustum = torch.stack((xs, ys, ds), -1)  # torch.Size([41, 8, 22, 3])
    return nn.Parameter(frustum, requires_grad=False)


def main():
    frustum = create_frustum()


if __name__ == '__main__':
    main()
