# AquaNet 损失函数与监督实施方案

**状态**：设计评审结论，作为后续实现依据

**更新日期**：2026-08-18

**适用范围**：当前 RoboNav/AquaNet、KinoGoal 数据转换产物与 `StreamingSequenceBPTTTrainLoop`

---

## 1. 文档目的

本文对原始多任务损失设计进行可实现性审查，并给出与当前代码、标签语义一致的计算方法和分阶段实施计划。

原始总损失为：

$$
L = \lambda_{traj}L_{traj}
  + \lambda_{env}L_{env}
  + \lambda_{depth}L_{depth}
  + \lambda_{occ}L_{occ}.
$$

这一结构可以保留，但不能把所有项按原描述直接实现。当前结论是：

1. 位置、朝向、速度、基础运动学一致性和 depth loss 可以基于现有标签实现。
2. 当前 future velocity 位于“当前帧 body frame”，不能按原设计再次乘 $R(\theta)$。
3. 20 步预测终点是 2 秒 horizon 的终点，通常不是整个 episode 的最终 goal；二者不能无条件绑定。
4. 当前地图基础设施足以先实现圆形 footprint 的 clearance/collision loss，但不足以宣称实现了真实多边形 footprint 的 continuous swept volume。
5. 当前 clearance 是 unsigned distance，在障碍物内部可能缺少有方向的梯度；signed distance field 是可靠环境损失的重要前置改进。
6. 原设计中的 $L_{occ}$ 没有定义对应的预测头、标签可见性和 unknown 语义，现阶段必须保持关闭。
7. `max_depth=5 m` 已被确定为硬截断。当前实现让超过 5 m 的有效深度以 5 m 参与监督，这与原文“超量程不参与回归”冲突；本文以当前硬截断行为为准。
8. 第一版使用固定权重和充分日志，不直接引入 GradNorm。

---

## 2. 当前数据与模型 contract

### 2.1 轨迹输出和标签

`TrajectoryHead` 和 `FutureTrajectoryTensorSmith` 的状态维度均为：

$$
s_k = [x_k, y_k, \sin\theta_k, \cos\theta_k, v_{x,k}, v_{y,k}, \omega_k],
\qquad k=1,\ldots,K,
$$

其中当前配置：

```text
K = 20
delta_t = 0.1 s
horizon = 2.0 s
```

坐标语义如下：

- $x/y$：未来位置在当前帧 body frame 下的坐标，单位 m。
- $\sin\theta/\cos\theta$：未来朝向相对于当前帧的 yaw。
- $v_x/v_y$：未来速度已经转换到当前帧 body frame，单位 m/s。
- $\omega$：平面 yaw rate，单位 rad/s。

特别需要注意：$v_x/v_y$ 不是“每个未来状态自身 body frame”中的速度。数据转换中的关系实际为：

$$
v_k^{current} = R_{current}^{T}v_k^{world}.
$$

在当前真实场景上，使用 $\Delta p/\Delta t \approx v^{current}$ 的平均残差约为：

```text
mean = 0.000073 m/s
p95  = 0.000476 m/s
```

如果再乘一次 $R(\theta_k)$，残差变为：

```text
mean = 0.0253 m/s
p95  = 0.0690 m/s
```

因此当前标签下的运动学损失不得再次旋转 $v_x/v_y$。

### 2.2 future trajectory 的尾部语义

当 episode 剩余帧数少于 20 时，converter 会重复最终状态补满 20 步。当前最终状态的位置、线速度和角速度均保持停车状态，因此这些 padding 表达的是“到达后保持静止”，不是无效元素。

第一版允许这些步骤参与监督。如果未来数据集使用不同 padding 规则，必须新增显式 `future_valid_mask`，不能仅依赖数值猜测有效性。

### 2.3 Goal 语义

`GoalTensorSmith` 输出：

$$
g = [x_g,y_g,\theta_g,v_{x,g},v_{y,g},\omega_g].
$$

它描述整个 episode 的最终状态，而不是固定 2 秒 horizon 的终点。真实样本首帧中：

```text
2 秒专家终点距离当前：0.69 m
episode 最终 goal 距离当前：5.66 m
二者位置差：5.05 m
```

因此，预测第 20 点始终应监督到第 20 个 expert future state；只有在 goal 已进入 horizon 时，才能额外施加完整 terminal-to-goal loss。

### 2.4 Navigation map 语义

当前 `NavigationMap2DTensorSmith` 已提供当前 body frame 下的局部地图：

```text
x range: [-1, 5] m
y range: [-3, 3] m
resolution: 0.05 m
occupancy channels: [unknown, free, occupied]
clearance: unsigned meters
traversability: bool
robot radius: 0.25 m
safety margin: 0.05 m
```

当前检查场景的 3500 个 expert future points 均位于该 crop 内。但预测轨迹仍可能越界，因此 environment loss 必须明确 unknown 和 out-of-bounds 的策略。

### 2.5 Depth 语义

Depth 是沿鱼眼相机射线的欧氏距离。当前目标被归一化为：

$$
y = \operatorname{clip}(d / d_{max},0,1), \qquad d_{max}=5\text{ m}.
$$

有效性条件为：

$$
m = m_{ego}\land \operatorname{isfinite}(d)\land(d>0).
$$

当前 $d>5$ 的像素仍然有效，但 target 被截断到 1。这是硬截断监督，不是超量程忽略。

---

## 3. 通用 reduction 与量纲规则

所有 loss 先按自身有效元素做 weighted mean，再乘任务权重：

$$
\operatorname{weighted\_mean}(e,w)
=\frac{\sum_i w_i e_i}{\max(\sum_i w_i,\epsilon)}.
$$

禁止先对无效元素填零再直接 `.mean()`，因为有效率变化会改变 loss 尺度。

建议先将不同物理量的残差除以固定尺度：

$$
\tilde e_x=e_x/s_{xy},\quad
\tilde e_v=e_v/s_v,\quad
\tilde e_\omega=e_\omega/s_\omega.
$$

第一版建议从以下可解释尺度开始，并通过数据统计确认：

```text
s_xy    = 1.0 m
s_v     = 1.0 m/s
s_omega = 1.0 rad/s
s_accel = 1.0 m/s^2
s_jerk  = 1.0 m/s^3
```

这些 scale 用于消除量纲，不等同于最终任务权重。

`AquaNet.train_step()` 当前会对一个输入 sequence 内各帧的总 loss 取平均，再执行一次 backward。各帧内部必须先完成自身的有效元素归一化，之后 sequence 平均才具有稳定含义。

---

## 4. 轨迹监督 $L_{traj}$

建议把轨迹损失拆成可独立记录和调权的子项：

$$
L_{traj}
=w_{xy}L_{xy}
+w_{yaw}L_{yaw}
+w_{unit}L_{unit}
+w_{vel}L_{vel}
+w_{kin}L_{kin}
+w_{smooth}L_{smooth}.
$$

第一版中每个子项都必须单独写入日志，不能只输出聚合后的 `loss_traj`。

### 4.1 位置损失

对每个时刻的 $x/y$ 使用 Smooth L1：

$$
e_{xy,k}=\frac{1}{2}\sum_{c\in\{x,y\}}
\operatorname{SmoothL1}
\left(\frac{\hat p_{k,c}-p^*_{k,c}}{s_{xy}};\beta_{xy}\right).
$$

时间权重推荐先使用最简单的 terminal boost：

$$
w_k=
\begin{cases}
w_{terminal}, & k=K,\\
1, & \text{otherwise}.
\end{cases}
$$

$$
L_{xy}=\frac{\sum_k m_k w_k e_{xy,k}}
{\max(\sum_k m_k w_k,\epsilon)}.
$$

这里的 terminal 是第 20 个 future state，不是 episode goal。`ADE` 和 `FDE` 作为 metric 记录，不替代逐时刻训练误差。

### 4.2 朝向损失

令：

$$
\hat q_k=[\widehat{\sin\theta_k},\widehat{\cos\theta_k}],
\qquad q_k^*=[\sin\theta_k^*,\cos\theta_k^*].
$$

方向余弦损失：

$$
L_{yaw}=\operatorname{mean}_k\left[
1-\frac{\hat q_k\cdot q_k^*}
{\max(\|\hat q_k\|_2,\epsilon)}
\right].
$$

单位圆约束：

$$
L_{unit}=\operatorname{mean}_k
\left(\|\hat q_k\|_2^2-1\right)^2.
$$

实现要求：

- `epsilon` 必须存在，避免初始预测接近零时除零。
- `w_unit` 应明显小于主朝向项，避免 quartic penalty 主导训练。
- 日志中分别记录 direction error 和 unit-norm error。
- 可额外记录角度 MAE，但 metric 中使用 `atan2` 不代表训练必须依赖 `atan2`。

如果 pilot 中 cosine normalization 导致梯度尖峰，可退回到对 sin/cos 的直接 Smooth L1；它同样处理周期性，且实现更稳定。

### 4.3 速度损失

$$
L_{vel}=\operatorname{mean}_{k,c}
\operatorname{SmoothL1}\left(
\frac{\hat u_{k,c}-u^*_{k,c}}{s_c};\beta_{vel}
\right),
$$

其中：

$$
u_k=[v_{x,k},v_{y,k},\omega_k].
$$

静止 mask 可由 target 定义：

$$
m_{stop,k}=
\mathbf 1\left(\sqrt{v_{x,k}^{*2}+v_{y,k}^{*2}}<v_{stop}ight)
\land\mathbf 1(|\omega_k^*|<\omega_{stop}).
$$

第一版必须分别记录：

```text
velocity_error_moving
velocity_error_stopped
terminal_velocity_error
stop_sample_ratio
```

仅“单独统计”不会改变梯度，也不能防止停车样本被掩盖。如果 pilot 证明 stop error 明显较差，再使用 stop weight：

$$
w_{vel,k}=1+(w_{stop}-1)m_{stop,k}.
$$

不要在没有统计证据前默认大幅重加权。

### 4.4 运动学一致性

#### 4.4.1 位置—速度一致性

当前标签和输出速度都位于当前帧 body frame，因此使用梯形积分：

$$
r_{p,k}=
\hat p_k-\hat p_{k-1}
-\frac{\Delta t}{2}(\hat v_k+\hat v_{k-1}).
$$

$$
L_{kin,pos}=\operatorname{mean}_k
\operatorname{SmoothL1}(r_{p,k}/s_{xy};\beta_{kin}).
$$

第一个 interval 使用当前观测状态：

```text
p_0 = [0, 0]
v_0 = current twist[:2]
```

禁止在当前 contract 下计算 $R(\theta_k)\hat v_k$。只有未来把标签改成“每个 future state 自身机体系速度”后，才允许恢复该公式。

#### 4.4.2 朝向—角速度一致性

为了避免 angle wrap，可以直接在单位圆上比较相邻旋转。将预测 $q$ 归一化后，计算：

$$
\sin\Delta\hat\theta_k
=\hat s_k\hat c_{k-1}-\hat c_k\hat s_{k-1},
$$

$$
\cos\Delta\hat\theta_k
=\hat c_k\hat c_{k-1}+\hat s_k\hat s_{k-1}.
$$

角速度给出的期望增量为：

$$
\Delta\theta_k^\omega
=\frac{\Delta t}{2}(\hat\omega_k+\hat\omega_{k-1}).
$$

然后比较：

$$
L_{kin,yaw}=\operatorname{mean}_k\left[
1-sin\Delta\hat\theta_k\sin\Delta\theta_k^\omega
-\cos\Delta\hat\theta_k\cos\Delta\theta_k^\omega
\right].
$$

第一个 interval 使用：

```text
q_0 = [0, 1]
omega_0 = current twist[2]
```

最终：

$$
L_{kin}=L_{kin,pos}+w_{kin,yaw}L_{kin,yaw}.
$$

### 4.5 平滑性、专家动态与机器人极限

预测动态量：

$$
a_k=\frac{v_k-v_{k-1}}{\Delta t},
\qquad
\alpha_k=\frac{\omega_k-\omega_{k-1}}{\Delta t},
$$

$$
j_k=\frac{a_k-a_{k-1}}{\Delta t}.
$$

优先比较预测与专家动态，而不是无条件压小动态：

$$
L_{expert\_dyn}
=\operatorname{SmoothL1}(\hat a-a^*)
+w_\alpha\operatorname{SmoothL1}(\hat\alpha-\alpha^*)
+w_j\operatorname{SmoothL1}(\hat j-j^*).
$$

这允许专家急刹车，不会把所有轨迹都推向平均平滑。

机器人极限使用 hinge penalty：

$$
L_{limit,a}=\operatorname{mean}\left[
\operatorname{ReLU}(\|\hat a\|-a_{max})^2
\right],
$$

角加速度和 jerk 同理。

当前仓库没有可靠的 $a_{max}$、$\alpha_{max}$、$j_{max}$ 参数，因此 limit penalty 暂缓。不能从训练数据最大值直接冒充硬件极限。

---

## 5. 环境监督 $L_{env}$

建议拆分为：

$$
L_{env}=w_{collision}L_{collision}
+w_{clearance}L_{clearance}
+w_{goal}L_{goal}
+w_{progress}L_{progress}.
$$

### 5.1 圆形 footprint 的 collision/clearance

当前地图提供中心到最近障碍物的 unsigned clearance $d(p)$。在圆形 footprint 假设下：

```text
r_robot = 0.25 m
m_safe  = 0.05 m
```

基础损失：

$$
L_{collision}=\operatorname{mean}
\left[\operatorname{ReLU}(r_{robot}-d(\hat p))^2\right],
$$

$$
L_{clearance}=\operatorname{mean}
\left[\operatorname{ReLU}(r_{robot}+m_{safe}-d(\hat p))^2\right].
$$

两者允许重叠，使真正 collision 获得更高惩罚；如果后续发现重复权重难以解释，可合并为单一 safety-distance loss。

### 5.2 轨迹和地图查询

预测位置已经位于地图的当前 body frame，可通过 `grid_sample` 对 clearance 做可微查询。实现时必须用单元测试固定以下约定：

- tensor 的第一空间轴对应 body $x$，第二空间轴对应 body $y$。
- `grid_sample` 的 grid 顺序与该非传统轴布局正确匹配。
- crop 边界和 `align_corners` 与 `NavigationMap2DTensorSmith` 一致。
- unknown、occupied、free 三个 occupancy channel 不得混淆。

环境 loss 的归一化单位是有效 trajectory/footprint query 数量，不是地图像素数量。

### 5.3 Swept volume

仅查询 20 个离散中心点可能漏掉相邻点之间的碰撞。第一版环境 loss 应在每对相邻状态之间插值，插值间距不大于地图分辨率或预设安全步长：

$$
p_{k,j}=(1-\tau_j)p_{k-1}+\tau_jp_k,
\qquad \tau_j\in[0,1].
$$

对于圆形 footprint，中心轨迹的连续采样加 clearance threshold 已覆盖侧面和横移扫掠区域。

对非圆形真实 footprint，需要定义 body-frame 边界采样点 $f_j$：

$$
p_{k,j}^{footprint}=p_k+R(\theta_k)f_j.
$$

当前数据只提供圆形半径，没有 footprint polygon，因此真实 footprint 版本暂缓。

### 5.4 Unsigned clearance 的风险

当前 `esdf.npy` 被数据 contract 限制为非负数，实际是 unsigned clearance。预测点深入 occupied 区域时，$d(p)$ 可能在一片区域内恒为零：loss 非零，但对位置的梯度可能为零，无法指出逃离障碍物的方向。

可靠方案是预计算 signed distance field：

```text
free space:     positive
obstacle inside: negative
boundary:       zero
```

在 signed SDF 就绪前可以实现基础 clearance loss，但必须将“障碍物内部梯度弱”列为已知限制，不能把它视为完整 collision optimizer。

### 5.5 Unknown 和 out-of-bounds

必须在实现前明确策略：

- occupied：始终不安全。
- out-of-bounds：建议按不安全处理，防止模型通过逃出 crop 规避 loss。
- unknown：安全系统中建议保守处理为不安全；如果用于训练可能造成过强约束，则需要显式 unknown mask，而不是把 unknown 当 free。

无论选择哪一种，都必须分别记录 query 比例：

```text
query_free_ratio
query_occupied_ratio
query_unknown_ratio
query_out_of_bounds_ratio
```

### 5.6 Goal terminal loss

完整 terminal goal loss 只能在 goal 位于 horizon 时启用：

$$
L_{goal}=m_{goal}\left(
L_{goal,xy}+w_{goal,yaw}L_{goal,yaw}+w_{goal,vel}L_{goal,vel}
\right).
$$

推荐 converter 显式输出：

```text
reaches_goal_within_horizon: bool
```

在该字段加入前，可以使用 expert horizon endpoint 是否等于 goal 的容差判断作为临时方案，但不得只根据预测结果决定 mask。

无条件 terminal-to-goal loss 被明确禁止，因为它会与第 20 个 expert state 的监督冲突。

### 5.7 Progress loss

定义经过量纲归一化的状态距离：

$$
D(s,g)=
w_p\frac{\|p-p_g\|_2}{s_{xy}}
+w_y\left(1-\cos(\theta-\theta_g)\right)
+w_v\frac{\|v-v_g\|_2}{s_v}
+w_\omega\frac{|\omega-\omega_g|}{s_\omega}.
$$

不建议要求每个预测时刻都单调接近 goal。推荐只约束 horizon endpoint 不显著差于专家：

$$
L_{progress}=\operatorname{ReLU}
\left(D(\hat s_K,g)-D(s_K^*,g)-m_{progress}\right).
$$

这样专家为了绕障而暂时远离 goal 时不会被额外惩罚。该项与完整轨迹 imitation 有一定重复，只有在 pilot 证明能改善 goal progress 且不提高 collision 时才保留。

---

## 6. Depth 监督 $L_{depth}$

### 6.1 多尺度 target

每个预测尺度先通过 valid-mask-weighted area interpolation 生成目标：

$$
m_s=\operatorname{AreaResize}(m),
$$

$$
d_s^*=\frac{\operatorname{AreaResize}(m\cdot d^*)}
{\max(m_s,\epsilon)}.
$$

每个尺度按有效像素比例归一化，然后对尺度取等权平均。当前 `MultiScaleDepthLoss` 已实现这一基础 contract。

### 6.2 Log-range loss

先把 normalized depth 恢复成米：

$$
\hat d=d_{max}\hat y,\qquad d^*=d_{max}y^*.
$$

为了避免预测接近零时的无限梯度，定义稳定 log-range：

$$
z(d)=\log(d+d_0),
$$

其中 $d_0$ 是小的物理距离下界，而不是无量纲 magic epsilon。然后：

$$
L_{depth,s}=\operatorname{weighted\_mean}
\left(
\operatorname{SmoothL1}(z(\hat d_s)-z(d_s^*);\beta_{log}),
m_s
\right).
$$

$$
L_{depth}=\frac{1}{|S|}\sum_{s\in S}L_{depth,s}.
$$

`d_0` 和 `beta_log` 必须配置并通过近距离梯度测试，不能直接对零调用 `log`。

### 6.3 5 m 截断策略

当前已确认的策略为硬截断：

```text
d <= 5 m: 使用真实深度
d > 5 m:  target = 5 m，仍参与监督
```

这与原设计“超量程不参与回归”不同。除非后续明确改变产品语义，否则 loss 必须遵守当前 target/valid mask，不得在 loss 内再次偷偷排除 $d>5$ 像素。

如果未来改成传感器只能确认“至少 5 m”的 censored 标签，应设计 one-sided loss，而不是简单把像素标 invalid。

### 6.4 Depth gradient consistency

在 log depth 上计算横向和纵向差分：

$$
L_{grad}=\operatorname{mean}
\left|\nabla z(\hat d)-\nabla z(d^*)\right|.
$$

相邻两个像素都有效时，该 edge 才有效：

$$
m_x(i,j)=m(i,j)m(i,j+1),
\qquad
m_y(i,j)=m(i,j)m(i+1,j).
$$

第一版不加入该项。第二阶段先只在最高分辨率输出上启用；确认收益后再考虑所有尺度，避免重复强化同一边缘并增加调参量。

---

## 7. Occupancy loss $L_{occ}$ 的状态

原设计只在总式中出现 $L_{occ}$，没有定义：

- occupancy prediction head；
- 预测坐标系、范围和分辨率；
- unknown/free/occupied 的训练语义；
- 相机可见区域 mask；
- 该 head 是否部署到 DLA。

当前 occupancy/clearance 是 ground-truth navigation map，可用于 $L_{env}$ 查询，但 AquaNet 没有 occupancy prediction head。因此：

$$
\boxed{\lambda_{occ}=0}
$$

直到上述 contract 被单独评审并补齐。

如果未来新增 occupancy auxiliary head，需要特别处理“全局地图包含相机不可观察区域”的问题。没有 visibility mask 时监督完整局部地图，可能迫使网络记忆或猜测遮挡区，意义不明确。

如果所谓 $L_{occ}$ 实际只是预测轨迹查询 occupancy 的 collision cost，则它属于 $L_{env}$，不应作为独立任务重复计权。

---

## 8. 总损失和权重策略

在 $L_{occ}$ 暂缓期间：

$$
L = \lambda_{traj}L_{traj}
  + \lambda_{env}L_{env}
  + \lambda_{depth}L_{depth}.
$$

第一版：

1. 以 $\lambda_{traj}=1$ 为锚点。
2. 所有子项先完成有效元素归一化和物理量 scale 归一化。
3. 初始权重只要求 loss 和共享层 gradient norm 处于可比较数量级，不要求数值完全相等。
4. 同时记录 raw loss、weighted loss、有效元素数量和关键分组 metric。
5. pilot 只用于诊断，不在同一轮同时改变多个权重。

建议日志至少包含：

```text
loss_traj_xy_raw / weighted
loss_traj_yaw_raw / weighted
loss_traj_unit_raw / weighted
loss_traj_vel_raw / weighted
loss_traj_kin_raw / weighted
loss_depth_raw / weighted
loss_collision_raw / weighted
loss_clearance_raw / weighted
loss_goal_raw / weighted
loss_progress_raw / weighted
num_valid_steps
num_valid_depth_pixels
num_valid_environment_queries
ADE / FDE / yaw_error / terminal_speed_error
```

共享梯度监控应选择固定的共享 backbone 参数或特征层，避免在不同 head 私有参数上比较得到误导结论。

### 8.1 GradNorm 的启用条件

第一版不使用 GradNorm。仅当固定权重 pilot 稳定复现以下问题时再评估：

- 某一任务持续使共享层梯度大一个数量级以上；
- loss 数值下降但另一任务指标稳定恶化；
- 分阶段 warm-up 仍不能缓解冲突。

GradNorm 在 BPTT 下会增加额外梯度计算、显存和实现复杂度，不应用来掩盖错误标签、坐标系或 mask。

---

## 9. 分阶段实施计划

### Phase 0：contract 和统计基线

在新增损失前完成：

1. 固定 trajectory tensor 的 7 维顺序、单位和当前帧坐标系语义。
2. 固定 $\Delta t=0.1$ s，并让未来数据集通过配置提供，而不是散落 magic number。
3. 统计位置、速度、角速度、加速度和 jerk 分布。
4. 统计 stop/moving 比例和 near-goal 比例。
5. 统计 expert future points 的 map crop 覆盖率。
6. 为每种 reduction 写解析测试，特别是 all-invalid 和 partial-valid 情况。

### Phase 1：最小可靠训练目标

实现并接入：

1. $L_{xy}$：20 步 Smooth L1 + terminal horizon weight。
2. $L_{yaw}$：方向损失；小权重 $L_{unit}$。
3. $L_{vel}$：$v_x/v_y/\omega$ Smooth L1。
4. stop/moving 分组 metric；暂不默认 stop reweight。
5. $L_{kin}$：当前坐标系下的梯形速度积分和 yaw-rate 一致性。
6. multi-scale log-range depth loss，保留 5 m 硬截断语义。
7. 固定权重、raw/weighted loss 和共享梯度范数日志。

Phase 1 不加入 environment、goal/progress、jerk limit、occupancy head 或 GradNorm。目标是先证明轨迹标签和基础几何监督能稳定收敛。

### Phase 2：安全与动态质量

在 Phase 1 稳定后依次加入，每次只加入一组：

1. 专家 acceleration/angular-acceleration/jerk matching。
2. 最高分辨率 depth gradient consistency。
3. 圆形 footprint 的离散 clearance/collision loss。
4. 相邻轨迹点插值形成的 swept-center loss。
5. 显式 `reaches_goal_within_horizon` 后的 gated terminal goal loss。
6. expert-relative progress margin。

Environment loss 上线前应优先把 unsigned clearance 升级为 signed distance field；如果暂时不升级，必须保留已知梯度限制并单独监控 collision case。

### Phase 3：有条件的高级能力

只有在对应数据和收益证据存在时考虑：

1. 真实 footprint polygon/boundary points 与姿态相关 swept volume。
2. 有明确可见性监督的 occupancy auxiliary head 和 $L_{occ}$。
3. 经过硬件确认的 acceleration/jerk limit penalty。
4. 分阶段 warm-up 仍无法解决多任务冲突时的 GradNorm。
5. 多尺度 depth gradient loss。

---

## 10. 明确暂缓或禁止的内容

以下内容不能在缺少前置条件时实现：

| 内容 | 当前决定 | 重新启用条件 |
|---|---|---|
| 在运动学损失中使用 $R(\theta)v$ | 禁止 | velocity 标签改为各 future-state body frame 并完成迁移测试 |
| 无条件把第 20 点拉到 episode goal | 禁止 | 仅对 `reaches_goal_within_horizon=True` 的样本启用 |
| 真实 footprint swept volume | 暂缓 | 提供正式 footprint polygon/边界采样和坐标 contract |
| acceleration/jerk 硬限制 | 暂缓 | 提供经机器人硬件确认的限制值 |
| $L_{occ}$ | 关闭 | 定义预测 head、visibility mask、unknown 语义和部署要求 |
| GradNorm | 暂缓 | 固定权重和 warm-up 已稳定证明存在 Pareto 冲突 |
| 所有尺度的 depth gradient loss | 暂缓 | 最高分辨率版本证明有收益且开销可接受 |
| 在 loss 中排除所有 $d>5$ 像素 | 禁止默认修改 | 产品重新决定超量程语义并同步 TensorSmith contract |

---

## 11. 测试与验收要求

每个新增 loss 至少覆盖：

1. prediction 等于 target 时 loss 为零或接近零。
2. 一个可手算的小 tensor 得到精确期望值。
3. batch、时间、像素 mask 的归一化不受无效元素数量影响。
4. all-invalid 输入返回可反向传播的零，而不是 NaN。
5. 所有应训练参数得到 finite gradient。
6. yaw 在 $-\pi/\pi$ 边界连续。
7. stop/moving 和 reaches-goal 分支均被覆盖。
8. 运动学测试能区分“直接使用 current-frame velocity”和错误的 $R(\theta)v$。
9. map 查询测试固定 x/y 轴、crop 边界、unknown 和 out-of-bounds 语义。
10. BPTT sequence 内所有帧的 loss 被平均，且只发生一次 optimizer step。

训练 pilot 的最低验收项：

- 每个 raw/weighted loss 有限；
- 总 gradient norm 有限；
- ADE/FDE、yaw、terminal velocity 和 depth 指标分别可见；
- environment 启用后 collision/unknown/out-of-bounds query ratio 可见；
- 关闭某个 $\lambda$ 时对应 head 不产生意外梯度；
- loss 计算仅存在于训练图，不改变 DLA 推理图。新增预测 head（如 occupancy head）必须单独进行 DLA 评审。

---

## 12. 最终决策摘要

第一版的核心原则是：

```text
先把已有标签表达的事实监督正确，
再增加物理一致性，
最后才增加地图安全约束和自动权重平衡。
```

优先级从高到低为：

1. 轨迹位置、朝向、速度监督正确且可解释。
2. 与当前坐标系一致的运动学约束。
3. 稳定的多尺度 log-range depth 监督。
4. 专家动态和平滑性。
5. signed-distance 支持下的圆形 footprint 环境损失。
6. gated goal/progress。
7. 有真实数据 contract 后的 footprint、occupancy 和 GradNorm。

任何高级 loss 都不能用来补偿错误坐标系、错误 horizon、缺失 mask 或未定义标签。
