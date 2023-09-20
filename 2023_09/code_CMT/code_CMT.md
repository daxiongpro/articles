# 代码解读——CMT：Cross Modal Transformer

之前两篇文章写了 CMT [论文解读](../CMT_paper/CMT_paper.md)和[环境安装](../env_CMT/env_CMT.md)。本文记录在 debug 过程中对代码的理解。

## 源代码解读

首先观察一下 `projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py` 文件的 model：

```python
model = dict(
    type='CmtDetector',
...
    pts_bbox_head=dict(
        type='CmtHead',
...
        transformer=dict(
            type='CmtTransformer',
...
        )))
```

模型使用的是 CmtDetector 类。其中，有关 Transforemer 的最主要的部分都在 CmtHead 类中。

## CmtHead

CmtHead 的代码入口为 forward_single() 方法。

### forward_single

```python
def forward_single(self, x, x_img, img_metas):
    """
    x 是激光雷达点云数据经过 backbone 后的 bev 特征图
        x: [bs c h w] x_pts (1, 512, 180, 180)
    x_img 是图像数据经过 backbone 后的 6 个环视特征图，叠加在一块
        x_img: = (6, 256, 40, 100)
        return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
    """
    ret_dicts = []
    x = self.shared_conv(x)  # (1, 256, 180, 180)
  
    # query 的随机点作为 anchor。 torch.Size([900, 3])
    reference_points = self.reference_points.weight
    # torch.Size([1, 900, 3])
    reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
    mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
  
    # -------------------重点开始---------------------------
    # 获取图像 positional embedding, torch.Size([6, 40, 100, 256])
    rv_pos_embeds = self._rv_pe(x_img, img_metas)
  
    # 获取点云 positional embedding, torch.Size([32400(180*180), 256])
    bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
  
    # 获取点云、图像的 query, torch.Size([1, 900, 256])
    bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
    query_embeds = bev_query_embeds + rv_query_embeds

    outs_dec, _ = self.transformer(
                        x,                       # torch.Size([1, 256, 180, 180]) （pts token）
                        x_img,                   # torch.Size([6, 256, 40, 100])  （img token）
                        query_embeds,            # torch.Size([1, 900, 256])      （query）
                        bev_pos_embeds,          # torch.Size([32400, 256])       （pts embed）
                        rv_pos_embeds,           # torch.Size([6, 40, 100, 256])  （img embed）
                        attn_masks=attn_mask     # None
                    )
    outs_dec = torch.nan_to_num(outs_dec)  # (6, 1, 900, 256)
    # -------------------重点结束---------------------------

    # (1, 900, 3)
    reference = inverse_sigmoid(reference_points.clone())
  
    ...
```

> 上述代码主要完成了初始化 embedding 的操作。然后将 点云在 BEV 下的特征(x)、图像环视特征(x_img)、两种模态的 embeding、query embedding 作为参数传入 transormer。

### 图像 PE

图像数据的 positional encoder 在 _rv_pe 方法中。其中 rv 代表 range view(对应 bev 代表 bird eye view)。

```python
def _rv_pe(self, img_feats, img_metas):
    BN, C, H, W = img_feats.shape
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
    coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
    coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (self.pc_range[3] - 1) / self.depth_num
    coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

    coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
    coords[..., :2] = coords[..., :2] * coords[..., 2:3]  # torch.Size([40, 100, 64, 4])
  
    imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
    imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)  # torch.Size([6, 4, 4])
    # torch.Size([6, 40, 100, 64, 4])
    coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
    # torch.Size([6, 40, 100, 64, 3])
    coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None, None, :]
    return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))  # torch.Size([6, 40, 100, 256])
```

> 上述代码中，输入是图像 feature map 和相机参数。其原理与论文 PETR 类似。最后一行，self.rv_embedding 是 MLP。

### 点云 PE

点云数据的 PE 直接在 single_forward() 方法中：

```python
self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim)) 
```

其中，bev_embedding 是 MLP。coords_bev() 方法是一个属性方法，目的是得到 BEV 空间的 anchor 点。

```python
@property
def coords_bev(self):
    cfg = self.train_cfg if self.train_cfg else self.test_cfg
    x_size, y_size = (
        cfg['grid_size'][1] // self.downsample_scale,
        cfg['grid_size'][0] // self.downsample_scale
    )
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = (batch_x + 0.5) / x_size
    batch_y = (batch_y + 0.5) / y_size
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
    coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
    return coord_base
```

上述代码的返回值是下面 pos2embed() 方法的输入参数 pos 。

pos2embed() 方法实现如下：

```python
def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
  
    # torch.Size([1, 900, 512])
    return posemb
```

> 上述代码中，pos 形状为 [32400, 2]，代表在 BEV feature map 上的点，每个点都有一个二维坐标。第二个参数 num_pos_feats 表示 embed 向量的长度，默认值为 128，实际传入的是 256。

### 图像 query 

图像 query 生成部分在 _rv_query_embed() 方法中。

```python
def _rv_query_embed(self, ref_points, img_metas):
    """
    ref_points: torch.Size([1, 900, 3])
    """
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
    lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
    imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
    imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

    # 公式 6
    ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])  # torch.Size([1, 900, 3])
  
    # 世界坐标系下 xyz1，和 lidars2imgs 转换到 图像坐标系。
    # torch.Size([1, 6, 900, 4])
    proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                torch.cat([
                                    ref_points,  # torch.Size([1, 900, 3])
                                    ref_points.new_ones(*ref_points.shape[:-1], 1)  # torch.Size([1, 900, 1])
                                    ],  dim=-1),
                                lidars2imgs)  # torch.Size([1, 6, 4, 4])
  
    proj_points_clone = proj_points.clone()
    z_mask = proj_points_clone[..., 2:3].detach() > 0
    proj_points_clone[..., :3] = proj_points[..., :3] / (proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6) 
    # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape) 
  
    mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
    mask &= z_mask.squeeze(-1)
    # torch.Size([64])
    coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (self.pc_range[3] - 1) / self.depth_num
    proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
    proj_points_clone = torch.cat([proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
    projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

    projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                    / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(self.pc_range[:3]))[None, None, None, :]
  
    rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
    rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
    return rv_embeds  # torch.Size([1, 900, 256])

```

### 点云 query

点云 query 生成部分在 _bev_query_embed() 方法中。这部分代码与点云 PE 部分完全一样。

```python
def _bev_query_embed(self, ref_points, img_metas):
    """
    ref_points: torch.Size([1, 900, 3])
    """
    bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
    return bev_embeds
```

## 后记

代码解读完整版看我的 [github](https://github.com/daxiongpro/CMT/tree/dev)。

## 日期

2023/09/20：文章撰写日期
