这份文档给出 position embedding（PE）的完整数学定义和实现路径。

目标是对 PE grid 中每一个 pixel 计算 6 个 channel：

[
\boxed{
[r_x,r_y,r_z,q_x,q_y,m_{\rm valid}]
}
]

最终 tensor：

[
\boxed{
PE\in\mathbb R^{1\times6\times H'\times W'}
}
]

其中：

* (r_x,r_y,r_z)：该 pixel 对应的 **body/self frame 下的单位视线方向**；
* (q_x,q_y)：这条 ray 与 body-frame ground plane 的交点坐标，经过 `pe_range` clamp 并归一化到 ([-1,1])；
* (m_{\rm valid})：该 ray 是否在正向上与 ground plane 存在有限交点。

整体计算链路为：

[
\boxed{
\text{PE pixel}
\rightarrow
\text{camera-frame ray}
\rightarrow
\text{body-frame ray}
\rightarrow
\text{ground intersection}
\rightarrow
\text{normalize + mask}
}
]

其中 PerspectiveCamera 和 FisheyeCamera 的区别，只应存在于：

[
\boxed{
\text{pixel}\rightarrow\text{camera ray}
}
]

这一步。

从 camera ray 得到之后，后面的数学完全统一。

---

# 1. 坐标系定义

坐标系必须首先固定，否则后面的所有公式即使形式正确，也可能得到完全错误的结果。

## 1.1 Camera frame (C)

采用 OpenCV camera convention：

[
(x_C,y_C,z_C)
=============

(\text{right},\text{down},\text{forward})
]

即：

```text
+x_C = image right
+y_C = image down
+z_C = camera forward
```

一条 camera-frame unit ray 记作：

[
r_C=
\begin{bmatrix}
r_{Cx}\
r_{Cy}\
r_{Cz}
\end{bmatrix},
\qquad
|r_C|_2=1.
]

---

## 1.2 Body/self frame (B)

机器人自身坐标系定义为：

[
(x_B,y_B,z_B)
=============

(\text{forward},\text{left},\text{up})
]

即：

```text
+x_B = forward
+y_B = left
+z_B = up
```

当前设计中 ground plane 为：

[
\boxed{z_B=0}
]

稍后会给出 ground 不在 (z=0) 时的一般形式。

---

# 2. Extrinsic 的数学 contract

整个实现最好统一规定：

[
\boxed{
{}^BT_C
}
]

即 extrinsic 必须表示：

> camera frame → body frame

写成：

[
{}^BT_C=
\begin{bmatrix}
R_{BC} & t_{BC}\
0 & 1
\end{bmatrix}.
]

因此任意 camera-frame point：

[
p_C
]

转换到 body frame：

[
\boxed{
p_B=R_{BC}p_C+t_{BC}
}
]

其中：

[
R_{BC}\in SO(3)
]

负责 camera → body 的旋转，而：

[
t_{BC}=
\begin{bmatrix}
t_x\
t_y\
t_z
\end{bmatrix}
]

就是 **camera optical center 在 body frame 中的位置**。

因此：

```python
T_bc = extrinsic
R_bc = T_bc[:3, :3]
t_bc = T_bc[:3, 3]
```

如果现有数据中的 extrinsic 已经是：

[
{}^BT_C
]

则：

> 不需要 invert，也不要再额外手工执行一次 `right/down/forward → forward/left/up` 的轴变换。

该轴关系已经包含在 (R_{BC}) 中。

---

## 2.1 如果原始 calibration 给的是相反方向

如果拿到的是：

[
{}^CT_B
]

即：

> body → camera

那么必须先求逆。

若：

[
{}^CT_B=
\begin{bmatrix}
R_{CB}&t_{CB}\
0&1
\end{bmatrix},
]

则：

[
\boxed{
R_{BC}=R_{CB}^T
}
]

以及：

[
\boxed{
t_{BC}=-R_{CB}^Tt_{CB}.
}
]

推荐把这种转换放在 calibration ingestion 阶段完成。

也就是说 PE builder 内部只接受一种明确 contract：

[
\boxed{\text{extrinsic always means camera → body}}
]

这样最不容易产生 silent coordinate bug。

---

# 3. PE resolution 与 downsampling

设原始图像分辨率为：

[
H\times W
]

PE downsample factor：

[
s=\text{pe_downsample_factor}.
]

则：

[
\boxed{
H'=\frac Hs,\qquad
W'=\frac Ws
}
]

例如：

```text
H = 384
W = 512
s = 2

→ H' = 192
→ W' = 256
```

---

# 4. Downsample 后的 intrinsic

原始 intrinsic：

[
K=
\begin{bmatrix}
f_x&0&c_x\
0&f_y&c_y\
0&0&1
\end{bmatrix}.
]

PE grid 对应的 intrinsic 定义为：

[
\boxed{
f_x'=\frac{f_x}{s},\qquad
f_y'=\frac{f_y}{s},
}
]

[
\boxed{
c_x'=\frac{c_x}{s},\qquad
c_y'=\frac{c_y}{s}.
}
]

也就是：

[
K'=SK
]

其中：

[
S=
\begin{bmatrix}
1/s&0&0\
0&1/s&0\
0&0&1
\end{bmatrix}.
]

对于 fisheye：

[
D=[k_1,k_2,k_3,k_4]
]

**完全不缩放**。

即：

```text
fx, fy, cx, cy → / s
k1, k2, k3, k4 → unchanged
```

---

# 5. 为什么直接 (K/s) 是正确的

PE grid 上位置：

[
(u',v')
]

对应原始 image coordinate：

[
(u,v)=(su',sv').
]

那么：

[
\frac{u-c_x}{f_x}
=================

# \frac{su'-c_x}{f_x}

\frac{u'-c_x/s}{f_x/s}.
]

因此：

[
(u',v',K')
]

和：

[
(su',sv',K)
]

产生完全相同的 normalized camera coordinate。

---

# 6. 与 ResNet stride-2 stem 的空间对齐

如果 RGB backbone 开头是：

```text
Conv 7×7
stride = 2
padding = 3
```

那么 output index (j) 对应的 receptive-field center 为：

[
u
=

# 2j-3+\frac{7-1}{2}

2j.
]

因此当：

[
s=2
]

时，PE index (j) 使用：

[
u=2j
]

恰好与对应 RGB feature 的 receptive-field center 对齐。

所以当前场景直接使用：

[
\boxed{
f_x,f_y,c_x,c_y\rightarrow /s
}
]

是合适的。

---

## 6.1 Half-pixel convention

如果未来 PE 不是与 strided convolution feature 对齐，而是和类似：

```text
cv2.resize
torch.interpolate
resize-based feature map
```

对齐，则需要重新考虑 pixel-center convention。

典型 half-pixel mapping 为：

[
u=(j+0.5)s-0.5.
]

相应 principal point 可能需要写成：

[
c_x'
====

\frac{c_x+0.5}{s}-0.5.
]

但：

> 当前 ResNet stride-2 stem 场景不要引入这个 offset。

---

# 7. 创建 PE pixel grid

定义：

[
u=0,\ldots,W'-1
]

[
v=0,\ldots,H'-1.
]

PyTorch：

```python
v, u = torch.meshgrid(
    torch.arange(H_pe, dtype=torch.float32),
    torch.arange(W_pe, dtype=torch.float32),
    indexing="ij",
)
```

于是：

```text
u.shape = [H', W']
v.shape = [H', W']
```

后续全部 vectorize。

---

# 8. PerspectiveCamera：pixel → camera-frame ray

PerspectiveCamera 假定为理想 pinhole，不额外考虑 distortion。

首先：

[
a=\frac{u-c_x'}{f_x'}
]

[
b=\frac{v-c_y'}{f_y'}.
]

camera frame 为：

```text
+x = right
+y = down
+z = forward
```

所以 pixel 对应未归一化 ray：

[
\tilde r_C=
\begin{bmatrix}
a\
b\
1
\end{bmatrix}.
]

然后单位化：

[
\boxed{
r_C=
\frac{
[a,b,1]^T
}{
\sqrt{a^2+b^2+1}
}
}
]

因此：

[
\boxed{
|r_C|_2=1.
}
]

代码：

```python
a = (u - cx_pe) / fx_pe
b = (v - cy_pe) / fy_pe

ray_c = torch.stack(
    [a, b, torch.ones_like(a)],
    dim=-1,
)

ray_c = ray_c / torch.linalg.vector_norm(
    ray_c,
    dim=-1,
    keepdim=True,
)
```

---

# 9. FisheyeCamera：pixel → camera-frame ray

这里使用 OpenCV fisheye distortion model：

[
D=[k_1,k_2,k_3,k_4].
]

与 Perspective 不同，fisheye 不应该简单构造：

[
[a,b,1]
]

然后直接当作 ray。

我们首先从 distorted pixel 恢复该 pixel 所代表的真实视线角度 (\theta)，然后**直接在单位球面上构造 ray**。

这是整个 fisheye branch 的核心。

---

# 10. Fisheye distorted normalized coordinate

首先：

[
x_d=\frac{u-c_x'}{f_x'}
]

[
y_d=\frac{v-c_y'}{f_y'}.
]

定义 distorted radial coordinate：

[
\boxed{
\rho_d=\sqrt{x_d^2+y_d^2}.
}
]

在这里：

[
\rho_d=\theta_d
]

即它对应 OpenCV fisheye distortion model 中的 distorted angle。

---

# 11. OpenCV fisheye distortion model

OpenCV fisheye model 为：

[
\boxed{
\theta_d
========

\theta
\left(
1
+k_1\theta^2
+k_2\theta^4
+k_3\theta^6
+k_4\theta^8
\right)
}
]

其中：

* (\theta_d)：已知，由 pixel 得到；
* (\theta)：真正的 camera ray 与 optical axis 的角度。

因此，我们需要反求：

[
\theta.
]

---

# 12. Fisheye inverse distortion：Newton iteration

定义：

[
f(\theta)
=========

\theta
\left(
1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8
\right)
-\theta_d.
]

展开：

[
f(\theta)
=========

\theta
+k_1\theta^3
+k_2\theta^5
+k_3\theta^7
+k_4\theta^9
-\theta_d.
]

导数为：

[
\boxed{
f'(\theta)
==========

1
+3k_1\theta^2
+5k_2\theta^4
+7k_3\theta^6
+9k_4\theta^8
}
]

Newton iteration：

[
\boxed{
\theta_{n+1}
============

## \theta_n

\frac{f(\theta_n)}
{f'(\theta_n)}
}
]

初始值直接使用：

[
\boxed{
\theta_0=\theta_d.
}
]

对于固定并正常标定的 camera，实际通常使用 5～10 次 iteration 即可。

例如固定：

```python
num_iterations = 8
```

由于 camera calibration 固定，PE 通常也只需要预计算一次，因此这里没有必要为了节省非常少的计算而复杂化实现。

---

# 13. 从 (\theta) 直接构造 spherical unit ray

这一步采用直接的单位球面表达。

首先定义 image-plane radial direction：

[
e_x=\frac{x_d}{\rho_d}
]

[
e_y=\frac{y_d}{\rho_d}.
]

也可以理解为：

[
e_x=\cos\phi,\qquad
e_y=\sin\phi
]

其中：

[
\phi=\operatorname{atan2}(y_d,x_d).
]

真正的 camera-frame ray：

[
\boxed{
r_C=
\begin{bmatrix}
\sin\theta,e_x\
\sin\theta,e_y\
\cos\theta
\end{bmatrix}
}
]

即：

[
\boxed{
r_{Cx}
======

\sin\theta
\frac{x_d}{\rho_d}
}
]

[
\boxed{
r_{Cy}
======

\sin\theta
\frac{y_d}{\rho_d}
}
]

[
\boxed{
r_{Cz}=\cos\theta.
}
]

这个表达有一个非常重要的性质：

[
\boxed{
|r_C|_2=1
}
]

因为：

[
\sin^2\theta(e_x^2+e_y^2)+\cos^2\theta
======================================

# \sin^2\theta+\cos^2\theta

1.

]

因此它直接给出具有清晰几何意义的 3D unit direction。

不需要先构造一个 pinhole-style `[a,b,1]` representation。

---

# 14. 为什么 fisheye 推荐直接 spherical ray

另一种常见写法是：

[
r=\tan\theta
]

然后构造：

[
[
r\cos\phi,
r\sin\phi,
1
]
]

再进行 normalization。

在正常前向范围内，它与 spherical ray 可以表达同一个方向。

但是这里没有必要绕过这个中间 representation。

我们已经直接求出了：

[
\theta
]

因此最自然、语义最明确的做法就是：

[
\boxed{
[
\sin\theta\cos\phi,
\sin\theta\sin\phi,
\cos\theta
]
}
]

它直接表达：

> 这条 ray 在 3D 单位球面上的真实方向。

因此 v3 将其作为 fisheye 的 canonical implementation。

---

# 15. Fisheye optical center 特殊情况

当：

[
\rho_d\approx0
]

即：

[
x_d\approx0,\qquad
y_d\approx0
]

时不能计算：

[
\frac{x_d}{\rho_d},
\qquad
\frac{y_d}{\rho_d}.
]

此时 pixel 就位于 optical axis。

直接定义：

[
\boxed{
r_C=
[0,0,1]^T
}
]

即可。

实现中：

```python
eps_center = 1e-8
non_center = rho_d > eps_center
```

---

# 16. Fisheye branch 的推荐 PyTorch 实现

```python
x_d = (u - cx_pe) / fx_pe
y_d = (v - cy_pe) / fy_pe

theta_d = torch.sqrt(x_d**2 + y_d**2)

theta = theta_d.clone()

for _ in range(8):
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4 * theta2
    theta8 = theta4**2

    f = (
        theta
        * (
            1.0
            + k1 * theta2
            + k2 * theta4
            + k3 * theta6
            + k4 * theta8
        )
        - theta_d
    )

    df = (
        1.0
        + 3.0 * k1 * theta2
        + 5.0 * k2 * theta4
        + 7.0 * k3 * theta6
        + 9.0 * k4 * theta8
    )

    theta = theta - f / df
```

然后直接构造 spherical ray：

```python
eps_center = 1e-8

non_center = theta_d > eps_center

radial_scale = torch.zeros_like(theta_d)

radial_scale[non_center] = (
    torch.sin(theta[non_center])
    / theta_d[non_center]
)

rx_c = x_d * radial_scale
ry_c = y_d * radial_scale
rz_c = torch.cos(theta)

rx_c = torch.where(
    non_center,
    rx_c,
    torch.zeros_like(rx_c),
)

ry_c = torch.where(
    non_center,
    ry_c,
    torch.zeros_like(ry_c),
)

rz_c = torch.where(
    non_center,
    rz_c,
    torch.ones_like(rz_c),
)

ray_c = torch.stack(
    [rx_c, ry_c, rz_c],
    dim=-1,
)
```

理论上：

[
|r_C|=1.
]

如希望清理浮点误差，也可以最后再执行一次：

```python
ray_c = ray_c / torch.linalg.vector_norm(
    ray_c,
    dim=-1,
    keepdim=True,
)
```

这只是 numerical cleanup，不改变数学定义。

---

# 17. Perspective / Fisheye 从这里开始完全合流

现在无论 camera model 是什么，都已经得到：

[
\boxed{
r_C(u,v)
}
]

并满足：

[
\boxed{
|r_C(u,v)|=1.
}
]

后续逻辑不得再区分 Perspective/Fisheye。

也就是说代码结构推荐为：

```python
if camera_type == "PerspectiveCamera":
    ray_c = compute_perspective_rays(...)

elif camera_type == "FisheyeCamera":
    ray_c = compute_fisheye_rays(...)

# Everything below is camera-model independent.
ray_b = transform_rays_to_body(ray_c, extrinsic)

qx, qy, valid = intersect_ground(
    ray_b,
    extrinsic,
)

pe = encode_position_embedding(
    ray_b,
    qx,
    qy,
    valid,
    pe_range,
)
```

camera model 的差异只停留在：

```text
pixel → ray_c
```

这是非常重要的 abstraction boundary。

---

# 18. Camera-frame ray → Body-frame ray

ray 是 direction vector。

因此只应用 rotation：

[
\boxed{
r_B=R_{BC}r_C
}
]

绝对不能写成：

[
R_{BC}r_C+t_{BC}.
]

translation 不作用在 direction vector 上。

如果 tensor layout 为：

```text
ray_c [H', W', 3]
```

则：

```python
ray_b = torch.einsum(
    "ij,hwj->hwi",
    R_bc,
    ray_c,
)
```

也等价于：

```python
ray_b = ray_c @ R_bc.T
```

因为这里最后一个维度存储的是 vector components。

得到：

[
r_B=
\begin{bmatrix}
r_x\
r_y\
r_z
\end{bmatrix}.
]

由于 (R_{BC}) 是 rotation：

[
\boxed{
|r_B|=|r_C|=1.
}
]

最终 PE 的前三个 channel 就是：

[
\boxed{
[r_x,r_y,r_z]
=============

[r_{Bx},r_{By},r_{Bz}]
}
]

也就是说：

> 六个 PE channel 的几何量全部统一在 body/self frame 中解释。

---

# 19. Camera optical center

camera origin 在 body frame 中就是：

[
\boxed{
o_B=t_{BC}
}
]

即：

[
o_B=
\begin{bmatrix}
o_x\
o_y\
o_z
\end{bmatrix}
=============

\begin{bmatrix}
t_x\
t_y\
t_z
\end{bmatrix}.
]

例如：

```text
camera:
forward offset = 0.2 m
left offset    = 0.0 m
height         = 0.55 m
```

则：

[
o_B=
[0.2,0,0.55]^T.
]

---

# 20. Body frame 中完整 ray equation

对于每一个 pixel：

[
\boxed{
p_B(\lambda)
============

o_B+\lambda r_B
}
]

其中：

[
\boxed{
\lambda\geq0
}
]

因为这里表示的是：

> ray

而不是：

> 无限延伸的 line。

展开：

[
x(\lambda)=o_x+\lambda r_x
]

[
y(\lambda)=o_y+\lambda r_y
]

[
z(\lambda)=o_z+\lambda r_z.
]

因为 (r_B) 是 unit vector，所以 (\lambda) 还有非常清楚的物理意义：

> 从 camera optical center 沿该视线方向前进的 metric distance。

---

# 21. 与 ground plane 求交

当前 ground plane：

[
\boxed{
z_B=0.
}
]

求交条件：

[
o_z+\lambda r_z=0.
]

因此：

[
\boxed{
\lambda_{\rm ground}
====================

-\frac{o_z}{r_z}.
}
]

然后：

[
\boxed{
q_x=o_x+\lambda_{\rm ground}r_x
}
]

[
\boxed{
q_y=o_y+\lambda_{\rm ground}r_y.
}
]

完整 3D intersection 为：

[
q_B
===

# o_B+\lambda_{\rm ground}r_B

[q_x,q_y,0]^T.
]

PE 中只保留：

[
(q_x,q_y).
]

---

# 22. Ground plane 的一般形式

当前使用：

[
z_B=0
]

但从数学上更一般地，如果真实 ground plane 是：

[
z_B=z_g
]

那么：

[
o_z+\lambda r_z=z_g.
]

因此：

[
\boxed{
\lambda_{\rm ground}
====================

\frac{z_g-o_z}{r_z}
}
]

当前方案只是取：

[
z_g=0
]

于是：

[
\lambda_{\rm ground}
====================

-\frac{o_z}{r_z}.
]

所以实现时可以明确保留：

```python
ground_z = 0.0
```

然后使用通式：

```python
lam = (ground_z - t_bc[2]) / rz
```

这样数学意义更加明确，也方便未来 body origin 定义发生变化时扩展。

---

# 23. `m_valid` 的严格定义

一个无限长的 3D line，只要：

[
r_z\neq0
]

就会与 (z=z_g) 平面相交。

但这里是 ray：

[
\lambda\geq0.
]

所以一个 pixel 的 ground intersection 有效，当且仅当：

1. ray 不平行于 ground；
2. intersection 位于 ray 的正方向。

即：

[
\boxed{
m_{\rm valid}
=============

\mathbf 1
\left[
|r_z|>\epsilon
\land
\lambda_{\rm ground}>0
\right].
}
]

例如：

[
\epsilon=10^{-6}.
]

---

# 24. 为什么必须检查 (\lambda>0)

假设 camera 高度：

[
o_z=0.5.
]

如果 ray 向下：

[
r_z=-0.5
]

那么：

[
\lambda
=======

# -\frac{0.5}{-0.5}

1>0.
]

所以：

```text
valid = 1
```

反之，如果：

[
r_z=+0.5
]

则：

[
\lambda
=======

# -\frac{0.5}{0.5}

-1.
]

虽然无限长直线确实穿过 ground，但 intersection 在 camera ray 的反方向。

所以：

```text
valid = 0
```

因此不能只检查：

```python
abs(rz) > eps
```

还必须检查：

```python
lam > 0
```

---

# 25. 正常相机安装时的直观理解

如果：

[
o_z>0
]

且：

[
z_g=0,
]

那么：

[
\lambda=-\frac{o_z}{r_z}>0
]

等价于：

[
r_z<0.
]

也就是说：

```text
ray 向下   → ground intersection valid
ray 水平   → invalid
ray 向上   → invalid
```

但是代码中不要 hard-code：

```python
valid = rz < 0
```

而应该保留通用几何定义：

```python
valid = non_parallel & (lam > 0)
```

这样未来 camera/body/ground 定义发生变化时逻辑仍然正确。

---

# 26. 数值安全的 ground intersection

不要直接：

```python
lam = -t_z / rz
```

然后再 mask。

因为接近 horizon：

[
r_z\approx0
]

的地方可能先生成：

```text
inf
NaN
```

推荐：

```python
eps = 1e-6

non_parallel = torch.abs(rz) > eps

safe_rz = torch.where(
    non_parallel,
    rz,
    torch.ones_like(rz),
)

lam = (
    ground_z - t_bc[2]
) / safe_rz

valid = (
    non_parallel
    & (lam > 0.0)
)
```

然后：

```python
qx = t_bc[0] + lam * rx
qy = t_bc[1] + lam * ry
```

对于 parallel ray，`lam` 的临时数值没有几何意义，但后面必然会被 `valid=False` mask 掉。

---

# 27. `pe_range`

定义：

```python
pe_range = (
    x_min,
    y_min,
    x_max,
    y_max,
)
```

其中坐标全部是 **body frame 中的 metric ground coordinates**。

首先得到真实 intersection：

[
(q_x,q_y).
]

然后分别 clamp：

[
\bar q_x
========

\operatorname{clip}
(q_x,x_{\min},x_{\max})
]

[
\bar q_y
========

\operatorname{clip}
(q_y,y_{\min},y_{\max}).
]

---

# 28. Normalize 到 ([-1,1])

对于 (x)：

[
\boxed{
\hat q_x
========

2
\frac{
\bar q_x-x_{\min}
}{
x_{\max}-x_{\min}
}
-1
}
]

对于 (y)：

[
\boxed{
\hat q_y
========

2
\frac{
\bar q_y-y_{\min}
}{
y_{\max}-y_{\min}
}
-1
}
]

因此：

[
x=x_{\min}\Rightarrow\hat q_x=-1
]

[
x=x_{\max}\Rightarrow\hat q_x=+1.
]

同理适用于 (y)。

最终 PE 中存储的是：

[
\boxed{
(\hat q_x,\hat q_y)
}
]

而不是 metric coordinates。

---

# 29. Clamp 不能改变 `m_valid`

这一点必须明确。

假设真实 intersection：

[
q_x=100\ {\rm m}
]

而：

```text
x_max = 5 m
```

这条 ray 仍然确实与 ground 相交。

因此：

```text
m_valid = 1
```

只是：

[
q_x
\overset{\rm clamp}{\longrightarrow}
5
]

最后：

[
\hat q_x=+1.
]

所以：

```text
m_valid:
    表示 ray 是否真实与 ground 相交

pe_range:
    表示我们希望编码的 ground coordinate 数值范围
```

二者不能混在一起。

禁止写成：

```python
valid = (
    geometric_valid
    & q_inside_pe_range
)
```

---

# 30. Invalid ray 的 (q_x,q_y) 必须最终置零

如果：

[
m_{\rm valid}=0
]

则定义：

[
\boxed{
\hat q_x=0,\qquad
\hat q_y=0.
}
]

因此最终：

```text
invalid pixel:
qx = 0
qy = 0
m  = 0
```

而：

```text
qx = 0
qy = 0
m  = 1
```

仍然可以明确表达：

> 这是一个有效的 ground intersection，只是归一化后的坐标恰好位于 0。

---

# 31. Invalid mask 必须在 normalization 之后应用

正确顺序：

```text
physical ground intersection
        ↓
clamp
        ↓
normalize to [-1, 1]
        ↓
invalid → qx=qy=0
```

即：

```python
qx = torch.clamp(qx, x_min, x_max)
qy = torch.clamp(qy, y_min, y_max)

qx = (
    2.0
    * (qx - x_min)
    / (x_max - x_min)
    - 1.0
)

qy = (
    2.0
    * (qy - y_min)
    / (y_max - y_min)
    - 1.0
)

qx = torch.where(
    valid,
    qx,
    torch.zeros_like(qx),
)

qy = torch.where(
    valid,
    qy,
    torch.zeros_like(qy),
)

m_valid = valid.to(torch.float32)
```

不要先：

```python
physical_q = 0
```

然后再 normalize。

因为除非 `pe_range` 关于零严格对称，否则 physical zero 并不会归一化成 PE zero。

---

# 32. 最终 6 个 channel

最终每个 pixel：

[
\boxed{
PE(u,v)
=======

[
r_{Bx},
r_{By},
r_{Bz},
\hat q_x,
\hat q_y,
m_{\rm valid}
]
}
]

其中：

[
r_{Bx},r_{By},r_{Bz}\in[-1,1]
]

因为它们来自 unit direction。

同时：

[
\hat q_x,\hat q_y\in[-1,1]
]

以及：

[
m_{\rm valid}\in{0,1}.
]

因此整个 encoding 的 numerical scale 非常统一。

---

# 33. Tensor shape contract

建议整个 pipeline 始终保持：

```text
u, v                [H', W']

ray_camera           [H', W', 3]

R_bc                 [3, 3]
t_bc                 [3]

ray_body             [H', W', 3]

lambda_ground        [H', W']

q_x                  [H', W']
q_y                  [H', W']

valid                [H', W']

PE                   [6, H', W']

final PE             [1, 6, H', W']
```

最后：

```python
pe = torch.stack(
    [
        rx,
        ry,
        rz,
        qx,
        qy,
        m_valid,
    ],
    dim=0,
).unsqueeze(0)
```

得到：

```text
[1, 6, H', W']
```

推荐：

```python
dtype = torch.float32
```

包括：

```python
m_valid
```

最终也存成：

```text
0.0 / 1.0
```

而不是 bool。

---

# 34. 推荐的软件结构

推荐把代码拆成三个清晰的几何阶段：

```python
ray_c = pixel_to_camera_ray(
    camera_type,
    intrinsics,
    distortion,
    ...
)

ray_b = camera_ray_to_body(
    ray_c,
    extrinsic,
)

qx, qy, valid = intersect_ground(
    ray_b,
    extrinsic,
    ground_z,
)

pe = encode_position_embedding(
    ray_b,
    qx,
    qy,
    valid,
    pe_range,
)
```

更加具体：

```python
if camera_type == "PerspectiveCamera":
    ray_c = compute_perspective_rays(...)

elif camera_type == "FisheyeCamera":
    ray_c = compute_fisheye_rays(...)

ray_b = transform_rays_to_body(
    ray_c,
    R_bc,
)

qx, qy, valid = intersect_ground(
    ray_b,
    t_bc,
    ground_z=0.0,
)

pe = encode_position_embedding(
    ray_b,
    qx,
    qy,
    valid,
    pe_range,
)
```

其中最重要的设计原则是：

[
\boxed{
\text{camera model difference ends at pixel → ray}
}
]

后面的 code path 完全共享。

---

# 35. 完整核心 PyTorch implementation skeleton

下面基本可以直接按照这个结构实现。

```python
def build_position_embedding(
    *,
    image_height,
    image_width,
    pe_downsample_factor,
    camera_type,
    fx,
    fy,
    cx,
    cy,
    extrinsic,          # [4, 4], camera -> body
    pe_range,           # (x_min, y_min, x_max, y_max)
    distortion=None,    # fisheye: (k1, k2, k3, k4)
    ground_z=0.0,
    device=None,
    dtype=torch.float32,
):
    s = pe_downsample_factor

    h_pe = image_height // s
    w_pe = image_width // s

    fx_pe = fx / s
    fy_pe = fy / s
    cx_pe = cx / s
    cy_pe = cy / s

    v, u = torch.meshgrid(
        torch.arange(
            h_pe,
            dtype=dtype,
            device=device,
        ),
        torch.arange(
            w_pe,
            dtype=dtype,
            device=device,
        ),
        indexing="ij",
    )

    # ------------------------------------------------------------
    # Pixel -> camera-frame unit ray
    # ------------------------------------------------------------

    if camera_type == "PerspectiveCamera":
        a = (u - cx_pe) / fx_pe
        b = (v - cy_pe) / fy_pe

        ray_c = torch.stack(
            [
                a,
                b,
                torch.ones_like(a),
            ],
            dim=-1,
        )

        ray_c = ray_c / torch.linalg.vector_norm(
            ray_c,
            dim=-1,
            keepdim=True,
        )

    elif camera_type == "FisheyeCamera":
        k1, k2, k3, k4 = distortion

        x_d = (u - cx_pe) / fx_pe
        y_d = (v - cy_pe) / fy_pe

        theta_d = torch.sqrt(
            x_d**2 + y_d**2
        )

        # Invert:
        #
        # theta_d =
        # theta * (
        #     1
        #     + k1 theta^2
        #     + k2 theta^4
        #     + k3 theta^6
        #     + k4 theta^8
        # )

        theta = theta_d.clone()

        for _ in range(8):
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta4 * theta2
            theta8 = theta4**2

            f = (
                theta
                * (
                    1.0
                    + k1 * theta2
                    + k2 * theta4
                    + k3 * theta6
                    + k4 * theta8
                )
                - theta_d
            )

            df = (
                1.0
                + 3.0 * k1 * theta2
                + 5.0 * k2 * theta4
                + 7.0 * k3 * theta6
                + 9.0 * k4 * theta8
            )

            theta = theta - f / df

        # Direct spherical unit ray:
        #
        # rx = sin(theta) * x_d / theta_d
        # ry = sin(theta) * y_d / theta_d
        # rz = cos(theta)

        eps_center = 1e-8

        non_center = theta_d > eps_center

        radial_scale = torch.zeros_like(
            theta_d
        )

        radial_scale[non_center] = (
            torch.sin(theta[non_center])
            / theta_d[non_center]
        )

        rx_c = x_d * radial_scale
        ry_c = y_d * radial_scale
        rz_c = torch.cos(theta)

        rx_c = torch.where(
            non_center,
            rx_c,
            torch.zeros_like(rx_c),
        )

        ry_c = torch.where(
            non_center,
            ry_c,
            torch.zeros_like(ry_c),
        )

        rz_c = torch.where(
            non_center,
            rz_c,
            torch.ones_like(rz_c),
        )

        ray_c = torch.stack(
            [rx_c, ry_c, rz_c],
            dim=-1,
        )

        # Optional numerical cleanup.
        ray_c = ray_c / torch.linalg.vector_norm(
            ray_c,
            dim=-1,
            keepdim=True,
        )

    else:
        raise ValueError(
            f"Unsupported camera type: {camera_type}"
        )

    # ------------------------------------------------------------
    # Camera ray -> body ray
    # ------------------------------------------------------------

    R_bc = extrinsic[:3, :3]
    t_bc = extrinsic[:3, 3]

    ray_b = torch.einsum(
        "ij,hwj->hwi",
        R_bc,
        ray_c,
    )

    rx = ray_b[..., 0]
    ry = ray_b[..., 1]
    rz = ray_b[..., 2]

    # ------------------------------------------------------------
    # Ground intersection
    #
    # p(lambda) = t_bc + lambda * ray_b
    #
    # z(lambda) = ground_z
    #
    # lambda =
    #     (ground_z - t_bc.z) / ray_b.z
    # ------------------------------------------------------------

    eps_ground = 1e-6

    non_parallel = (
        torch.abs(rz) > eps_ground
    )

    safe_rz = torch.where(
        non_parallel,
        rz,
        torch.ones_like(rz),
    )

    lam = (
        ground_z - t_bc[2]
    ) / safe_rz

    valid = (
        non_parallel
        & (lam > 0.0)
    )

    qx = (
        t_bc[0]
        + lam * rx
    )

    qy = (
        t_bc[1]
        + lam * ry
    )

    # ------------------------------------------------------------
    # Clamp + normalize ground coordinate
    # ------------------------------------------------------------

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = pe_range

    qx = torch.clamp(
        qx,
        x_min,
        x_max,
    )

    qy = torch.clamp(
        qy,
        y_min,
        y_max,
    )

    qx = (
        2.0
        * (qx - x_min)
        / (x_max - x_min)
        - 1.0
    )

    qy = (
        2.0
        * (qy - y_min)
        / (y_max - y_min)
        - 1.0
    )

    # Invalid ground rays carry no ground-coordinate information.
    qx = torch.where(
        valid,
        qx,
        torch.zeros_like(qx),
    )

    qy = torch.where(
        valid,
        qy,
        torch.zeros_like(qy),
    )

    m_valid = valid.to(dtype)

    # ------------------------------------------------------------
    # Final PE
    #
    # [rx_body, ry_body, rz_body, qx, qy, valid]
    # ------------------------------------------------------------

    pe = torch.stack(
        [
            rx,
            ry,
            rz,
            qx,
            qy,
            m_valid,
        ],
        dim=0,
    ).unsqueeze(0)

    # [1, 6, H_pe, W_pe]
    return pe
```

---

# 36. 一个可以手算的 ground-intersection unit test

假设：

[
o_B=[0,0,1]
]

即 camera 高 1 m。

一条 body-frame unit ray：

[
r_B
===

\frac1{\sqrt2}
[1,0,-1].
]

则：

[
r_z=-\frac1{\sqrt2}.
]

因此：

[
\lambda
=======

# -\frac1{-1/\sqrt2}

\sqrt2.
]

所以：

[
q_x
===

0
+
\sqrt2
\frac1{\sqrt2}
==============

1
]

[
q_y=0.
]

最终：

[
\boxed{
q=(1,0).
}
]

这个 case 非常适合直接写成 unit test。

---

# 37. Sanity check 1：camera optical axis

假设 camera optical axis 与 robot 正前方完全对齐。

camera center pixel：

[
r_C=[0,0,1].
]

经过 camera → body extrinsic 之后应该得到：

[
\boxed{
r_B\approx[1,0,0].
}
]

如果仍然得到：

```text
[0, 0, 1]
```

很可能说明 camera → body rotation 没有应用。

---

# 38. Sanity check 2：image right / left

camera frame：

```text
+x_C = image right
```

body frame：

```text
+y_B = left
```

对于典型 forward-facing camera：

```text
image right
→ body right
→ r_By < 0
```

因此：

[
\boxed{
\text{image right}\Rightarrow r_{By}<0
}
]

而：

[
\boxed{
\text{image left}\Rightarrow r_{By}>0.
}
]

这是检查左右轴方向非常有效的方法。

---

# 39. Sanity check 3：image lower / upper

camera：

```text
+y_C = down
```

body：

```text
+z_B = up
```

对于正常安装：

```text
image lower part
→ ray points more downward
→ r_Bz becomes more negative
→ more likely to hit ground
```

因此通常下半幅大量 pixel：

[
r_z<0
]

并：

```text
m_valid = 1
```

上半幅则可能：

[
r_z>0
]

于是：

```text
m_valid = 0
```

具体 horizon 位置由 camera extrinsic 决定。

---

# 40. Sanity check 4：horizon

如果：

[
r_z=0
]

则 ray 与 ground plane 平行。

必须得到：

```text
m_valid = 0
qx = 0
qy = 0
```

不能出现：

```text
NaN
inf
```

---

# 41. Sanity check 5：ray norm

Perspective 和 Fisheye 得到 camera ray 后，都应满足：

[
|r_C|_2\approx1.
]

经过 extrinsic 后：

[
|r_B|_2\approx1.
]

建议直接测试：

```python
torch.linalg.vector_norm(
    ray_b,
    dim=-1,
)
```

整张图应非常接近：

```text
1.0
```

---

# 42. Sanity check 6：可视化 (r_z)

将：

[
r_{Bz}
]

画成 heatmap。

正常 forward-facing / slightly downward-facing camera 通常应看到：

* image upper region：(r_z) 较大；
* 越往 lower region：(r_z) 越负；
* horizon 附近经过 (r_z=0)。

如果图像毫无这种几何结构，需要优先检查 extrinsic direction。

---

# 43. Sanity check 7：可视化 `m_valid`

将：

```python
m_valid
```

直接显示成 binary image。

典型情况下应该有明显的 horizon-like boundary：

```text
upper image

00000000000000
00000000000000
00000000000000
00111111111100
11111111111111
11111111111111

lower image
```

实际形状由 camera pitch、height 和 fisheye geometry 决定，不一定是一条直线。

如果全图都是：

```text
0
```

或者全图都是：

```text
1
```

优先检查：

1. extrinsic direction；
2. camera/body axis convention；
3. ground plane height；
4. ray transformation。

---

# 44. Sanity check 8：BEV scatter

对所有：

```python
valid == True
```

的 pixel，保存 normalization 之前的 metric：

[
(q_x,q_y)
]

然后画 BEV scatter。

应该看到一个从 camera/robot 周围向前展开的视场 footprint。

Perspective 通常比较接近扇形。

Fisheye 则会具有更大的 angular coverage。

如果 scatter：

* 整体翻转；
* 朝机器人后方；
* 左右颠倒；
* 大量无规则发散；

优先检查 extrinsic 和 camera axis convention。

---

# 45. 最终 mathematical contract

对于 PE grid 上的每个 pixel：

[
i=(u,v)
]

首先：

[
\boxed{
r_C(i)
======

\pi^{-1}
(u,v;K',D)
}
]

其中：

* Perspective：标准 pinhole inverse projection；
* Fisheye：inverse distortion 得到 (\theta)，再直接构造 spherical unit ray。

对于 fisheye：

[
\boxed{
\theta_d
========

\sqrt{
\left(\frac{u-c_x'}{f_x'}\right)^2
+
\left(\frac{v-c_y'}{f_y'}\right)^2
}
}
]

并求解：

[
\boxed{
\theta_d
========

\theta
(
1+k_1\theta^2
+k_2\theta^4
+k_3\theta^6
+k_4\theta^8
)
}
]

然后：

[
\boxed{
r_C=
\begin{bmatrix}
\sin\theta,x_d/\theta_d\
\sin\theta,y_d/\theta_d\
\cos\theta
\end{bmatrix}.
}
]

再执行：

[
\boxed{
r_B(i)=R_{BC}r_C(i)
}
]

camera origin：

[
\boxed{
o_B=t_{BC}
}
]

ground plane：

[
\boxed{
z=z_g
}
]

当前：

[
z_g=0.
]

intersection parameter：

[
\boxed{
\lambda_i
=========

\frac{
z_g-o_{B,z}
}{
r_{B,z}(i)
}
}
]

valid：

[
\boxed{
m_i
===

[
|r_{B,z}(i)|>\epsilon
]
[
\lambda_i>0
]
}
]

ground point：

[
\boxed{
q_i
===

o_B+\lambda_i r_B(i)
}
]

取：

[
(q_{i,x},q_{i,y}).
]

clamp + normalization：

[
\boxed{
\hat q_x
========

2
\frac{
\operatorname{clip}
(q_x,x_{\min},x_{\max})
-x_{\min}
}{
x_{\max}-x_{\min}
}
-1
}
]

[
\boxed{
\hat q_y
========

2
\frac{
\operatorname{clip}
(q_y,y_{\min},y_{\max})
-y_{\min}
}{
y_{\max}-y_{\min}
}
-1
}
]

然后：

[
\boxed{
m_i=0
\Rightarrow
\hat q_{i,x}
============

# \hat q_{i,y}

0.

}
]

最终：

[
\boxed{
PE_i
====

[
r_{Bx},
r_{By},
r_{Bz},
\hat q_x,
\hat q_y,
m_i
]
}
]

最终 tensor：

[
\boxed{
PE
\in
\mathbb R^{1\times6\times H'\times W'}.
}
]

---

# 46. 最终实现时最值得盯住的 8 个点

1. **Extrinsic contract 必须固定为 camera → body。**

2. **Direction vector 只乘 (R)，绝不加 (t)。**

3. **PE 与 stride-based feature 对齐时，intrinsic 直接除以 (s)。**

4. **Fisheye distortion coefficients (k_1,k_2,k_3,k_4) 不随 resolution 缩放。**

5. **Fisheye inverse 得到 (\theta) 后，直接构造 spherical unit ray，不经过 `[a,b,1]` 的 pinhole intermediate representation。**

6. **Ground intersection 处理的是 ray，不是 infinite line，因此必须有 (\lambda>0)。**

7. **`m_valid` 只代表 geometric ground intersection，与 intersection 是否位于 `pe_range` 内无关。**

8. **Invalid (q_x,q_y) 必须在 clamp + normalization 完成之后最终置零。**

如果这 8 条 contract 在代码中保持不变，那么整个 6-channel PE：

[
\boxed{
[
r_x,r_y,r_z,q_x,q_y,m_{\rm valid}
]
}
]

就具有完整、统一而且没有 mixed-frame 歧义的几何语义。
