可以。你现在已经越过“把仿真平台跑起来”这一步，下一阶段应该正式把 **DE-P 从 UAV trajectory network 改造成 Ackermann ground-vehicle network**。这里最重要的不是简单换输入，而是同时重构 **传感器表示、状态表示、候选轨迹参数化、损失函数和训练顺序**。

我重新看了你 V4.9.1 分支里的网络实现。当前 `DepNetwork` 的核心仍然是：单通道深度图经过 MobileNetV3 backbone，得到大约 `[B,64,3,5]` 的特征；9D 状态经过 `StateTransform` 后按 3×5 primitive 展开，再和图像特征拼接；`DepHead` 对每个 lattice cell 输出 9 个轨迹终态参数和一个 score，因此天然产生 3×5=15 条候选。 `StateTransform` 中这 9D 输入具体还是 UAV 的 `v_xyz + a_xyz + goal_xyz`，输出则是 UAV 的 `p/v/a` terminal state。

所以我建议 **保留 V4.9.1 的“多候选 + score + differentiable safety training”思想，但不要保留它的 3×5 空间卷积语义和 PVA 输出形式。**

---

# 一、先把新的 DE-P-Car 网络定义清楚

我建议第一版正式网络命名为：

```text
DEPCarNetV1
```

整体结构变成：

```text
                 Depth Camera
                [B,1,96,160]
                       │
                       ▼
            MobileNetV3 Depth Encoder
                       │
                  F_depth
                       │
                       │
VLP-16 ──► LiDAR BEV Encoder ──► F_lidar
                       │
                       │
Vehicle + Route State ─► State MLP
                       │
                       ▼
             Multimodal Fusion
                       │
                       ▼
          15 Ackermann Candidate Queries
              3 speed × 5 curvature
                       │
                       ▼
             Ackermann Candidate Head
                       │
        ┌──────────────┴─────────────┐
        ▼                            ▼
 trajectory parameters            score
        │
        ▼
 Differentiable Bicycle Rollout
        │
        ▼
x(t),y(t),yaw(t),v(t),steering(t)
        │
        ▼
 swept-footprint / guidance / kinematic loss
```

也就是说，V4.9.1 中的：

```text
Depth Backbone
+
State
+
3×5 Candidate Head
+
Score
```

四个思想继续保留。

但物理含义全部换成 Ackermann。

---

# 二、深度相机和 LiDAR 不建议直接“拼成两个通道”

最简单的方法当然是：

```text
camera depth
+
lidar projected depth
↓
[B,2,96,160]
```

然后直接送 MobileNet。

我不建议把这个作为正式方案，因为这样会浪费 VLP-16 最大的优势：

> **360°视场。**

D435 只看前方，而 LiDAR 能看到：

```text
左侧
右侧
左后
右后
车尾
```

这些恰恰对 Ackermann 的转弯和恢复很重要。

所以我建议采用 **双编码器，中层融合**。

---

# 三、Depth Branch：这是最适合继承 V4.9.1 权重的部分

你现在的 V4.9.1 `DepBackbone` 本身就是针对：

```text
[B,1,96,160]
```

单通道深度设计的 MobileNetV3-Small，最后压缩成低分辨率特征。

因此 Car 版本可以保留：

```text
DepthEncoderV1
```

输入：

```text
RealSense depth
↓
clip invalid
↓
metric normalization
↓
resize 96×160
↓
[B,1,96,160]
```

然后：

```text
MobileNetV3
```

这里甚至可以尝试用 **V4.8.3 frozen checkpoint 的 depth backbone 做初始化**。

但是：

```text
只迁 backbone
```

不要迁：

```text
state_transform
DepHead
UAV primitive
trajectory output
score head
```

因为这些已经是 UAV-specific。

比较合理的实验是：

```text
A: Depth backbone random init
B: V4.8.3 backbone init
```

做一次小规模 A/B。

如果 B 收敛更快就保留 transfer learning；如果地面视角 domain shift 太大，就回到随机初始化。

不要先假设旧 backbone 一定更好。

---

# 四、LiDAR Branch：我建议用 BEV，而不是直接 PointNet

对于地面 Ackermann 小车，LiDAR 最自然的表示不是 3D point list，而是：

```text
Bird's-Eye View
```

因为你的轨迹本身就在：

[
x-y
]

平面运动。

建议：

```text
/velodyne_points
        │
        ▼
transform → base_link
        │
        ▼
crop local region
        │
        ▼
BEV rasterization
        │
        ▼
[B,C,H,W]
```

第一版 BEV 可以包含 4～6 个通道，例如：

```text
occupancy / point density
minimum height
maximum height
nearest range
visibility
validity
```

然后：

```text
LidarBEVEncoderV1
```

用一个非常轻量的 CNN。

不用一开始上复杂 PointTransformer。

你的目标不是 LiDAR 语义分割，而是：

> **判断未来车辆 swept footprint 周围有没有空间。**

BEV 对这个问题非常合适。

---

# 五、真正推荐的融合方式：Candidate-conditioned Fusion

这里我不建议直接把：

```text
F_depth
F_lidar
state
```

简单 concat 后再输出 `[B,?,3,5]`。

原因是一个很容易忽略的问题：

### UAV 的 3×5 和图像空间是对齐的

原来 3×5 是：

```text
pitch × yaw
```

所以深度图：

```text
左上
中上
右上
...
```

和 primitive 方向大体有对应关系。

因此 V4.9.1 用：

```text
[B,C,3,5]
```

直接输出 15 个格子很合理。

---

### Car 的 3×5 不再是图像空间

现在你希望它表示：

```text
speed × curvature
```

例如：

```text
          hard-L   left   straight   right   hard-R

low         ●       ●       ●         ●       ●
mid         ●       ●       ●         ●       ●
high        ●       ●       ●         ●       ●
```

这里第一行、第二行、第三行不再代表图像上中下。

而是：

```text
低速
中速
高速
```

所以如果还强制：

```text
CNN feature map [3,5]
=
candidate lattice [3,5]
```

会产生一个假的空间对应关系。

这次最好把它修掉。

---

# 六、因此我推荐新的 15 Candidate Query 结构

先定义：

```text
3 个 speed embedding
5 个 curvature embedding
```

组合出：

[
3\times5=15
]

个 candidate query。

例如：

```text
Q(i,j)
=
E_speed(i)
+
E_curvature(j)
```

每个 query 都表示：

> “如果我以这一档速度、这一档曲率通过当前环境，这条轨迹应该怎样微调？”

然后每个 query 去读取：

```text
Depth feature
LiDAR feature
Vehicle state
Local path state
```

最终输出自己的 trajectory residual + score。

这才是真正适配 Ackermann 的：

```text
3×5 lattice
```

它现在是一个**逻辑候选格**，不再强行等同于图像的 3×5 feature map。

---

# 七、甚至可以让 LiDAR 特征按照候选轨迹采样

这个设计非常适合你现在这个课题。

假设有一个基础 candidate：

```text
mid-speed + left
```

根据 bicycle model 能先得到一个 canonical path：

```text
      /
     /
----/
```

然后直接在 LiDAR BEV feature map 上沿这条轨迹采样：

```text
F1 F2 F3 F4 F5 ... Fn
```

进行 pooling。

于是：

```text
left candidate
```

主要看到左侧通道；

```text
right candidate
```

主要读取右侧；

```text
straight candidate
```

主要看正前方。

我建议把这个模块叫：

```text
TrajectoryConditionedBEVSamplerV1
```

它会比：

```text
global pooling LiDAR feature
```

更符合轨迹规划问题。

---

# 八、Depth branch 怎么和 Candidate 对齐？

深度图前视特征可以采用两种方式。

第一版可以简单：

```text
Depth backbone
↓
global + horizontal-region pooling
```

按照 candidate curvature：

```text
hard left
left
straight
right
hard right
```

重点读取对应的水平区域。

之后如果发现性能受限，再做：

```text
candidate query cross-attention
```

没有必要第一版直接上大 Transformer。

因此网络仍然可以非常轻：

```text
MobileNetV3 depth branch
+
small LiDAR BEV CNN
+
candidate-wise MLP
```

RTX 5070 Ti 跑训练和 ROS inference 都不会构成主要负担。

---

# 九、车辆状态我建议重新定义成 9D

这一点可以保留你现在 `observation_dim=9` 的代码习惯，但物理意义全部换掉。

我建议：

[
s=
[
v,,
a,,
\delta,,
\dot\psi,,
g_x,,
g_y,,
\sin e_\psi,,
\cos e_\psi,,
\kappa_{ref}
]
]

即：

```text
v              当前纵向速度
a              当前纵向加速度
δ              当前前轮转角
yaw_rate       当前转弯角速度

gx, gy         local subgoal在body frame中的位置

sin(eψ)
cos(eψ)        当前车头和参考路径航向误差

κref           Hybrid A*局部路径参考曲率
```

这样仍然是：

```text
[B,9]
```

但是比：

```text
vx vy vz ax ay az gx gy gz
```

更适合 Ackermann。

原 DE-P 的 9D 状态会针对每个 UAV primitive 做旋转变换。

Car 版本不需要这一步。

改成：

```text
StateEncoder
9
↓
32/64 dim
```

然后把状态 embedding 提供给所有 15 个 candidate query。

---

# 十、网络不要再直接输出 UAV 的 9D terminal PVA

原网络每个格子输出：

```text
px py pz
vx vy vz
ax ay az
score
```

`DepHead` 也明确固定为：

```text
9 trajectory channels + 1 score
```

Car 版本必须换掉。

我建议第一版每个 candidate 输出：

```text
Δδ_mid
Δδ_end
Δv_end
ΔT
score
```

也就是 5 个量。

其中 canonical lattice 本身已经提供：

```text
speed_anchor
curvature_anchor
```

网络只学习 residual。

---

# 十一、为什么我不推荐直接预测 steering command

不要做：

```text
network
↓
steering = 0.27
throttle = 0.42
```

因为这样会把 DE-P 最重要的优势丢掉：

> 同一时刻显式比较多条未来轨迹。

更合理的是：

```text
candidate anchor
+
network residual
↓
target steering profile
target velocity profile
trajectory duration
↓
differentiable bicycle rollout
```

例如：

[
\delta(0)
\rightarrow
\delta_{mid}
\rightarrow
\delta_{end}
]

同时：

[
v(0)\rightarrow v_{end}
]

在 (T) 秒内平滑变化。

然后积分：

[
\dot x=v\cos\psi
]

[
\dot y=v\sin\psi
]

[
\dot\psi=\frac{v}{L}\tan\delta
]

得到：

```text
x(t)
y(t)
yaw(t)
v(t)
δ(t)
```

整个 rollout 用 PyTorch 写成 differentiable graph。

这样 loss 才能真正反向传播进网络。

---

# 十二、这一层应该成为 Car 版本的 `StateTransform`

原来：

```text
StateTransform.pred_to_endstate()
```

负责：

```text
网络归一化输出
→ UAV primitive frame
→ body frame PVA
```

Car 版本建议直接废弃这套语义，新建：

```text
AckermannRolloutV1
```

变成：

```text
network candidate params
        │
        ▼
bound controls
        │
        ▼
bicycle integration
        │
        ▼
vehicle trajectory
```

这是整个 DE-P-Car 网络最关键的一次重构。

---

# 十三、训练依然不要变成“模仿 Hybrid A*”

你原来的 DE-P 一个很好的思想是：

> 网络不一定需要一个唯一正确的 GT trajectory。

而是：

```text
网络生成轨迹
↓
用环境authority算cost
↓
loss反传
```

这个思想应该继续。

所以 Hybrid A* 在训练中应该主要负责：

```text
提供参考方向/局部路径
```

而不是：

```text
Hybrid A* steering
→ 当成唯一GT
→ MSE模仿
```

否则你的网络最后只是一个 Hybrid A* policy compression。

---

# 十四、Static Safety Loss 应彻底改成 Swept Footprint

这部分比网络结构本身还重要。

不能再用：

```text
车辆中心点
```

判断碰撞。

建议训练时离线知道完整 map authority：

```text
occupancy map
↓
2D distance field
```

然后把车辆 footprint 近似成 5～7 个 circle：

```text
 front
   ○
   ○
   ○
   ○
   ○
 rear
```

每一个预测时刻：

```text
x(t),y(t),yaw(t)
```

把这些 circle center 转换到世界系。

查询：

[
D(x,y)
]

得到：

[
d_{\min}
========

\min_{t,k}
D(p_k(t))-r_k
]

训练：

```text
collision
near collision
clear
```

都可以形成连续梯度。

这才是真正的：

```text
Ackermann Static Safety Loss
```

---

# 十五、总体 loss 我建议重新整理为五部分

可以定义：

[
L=
\lambda_sL_{safe}
+
\lambda_gL_{guide}
+
\lambda_kL_{kin}
+
\lambda_cL_{comfort}
+
\lambda_dL_{diversity}
+
\lambda_rL_{score}
]

其中：

### `L_safe`

车辆整个 swept footprint 和静态障碍的 clearance。

---

### `L_guide`

与 Hybrid A* local path 的：

```text
cross-track error
heading error
progress
```

例如：

[
L_g=
\lambda_y e_y^2+
\lambda_\psi e_\psi^2-
\lambda_p\Delta s
]

---

### `L_kin`

即使 bicycle rollout 已经保证非完整约束，仍然检查：

```text
steering angle
steering rate
acceleration
speed
lateral acceleration
```

例如：

[
a_y=v^2\kappa
]

不能太大。

---

### `L_comfort`

控制：

```text
steering rate
steering acceleration
longitudinal jerk
```

避免车疯狂左右打方向。

---

### `L_diversity`

这一项非常重要。

确保 15 candidates 不会全部收敛成：

```text
15条几乎一样的直线
```

可以限制每个 candidate 对 anchor 的 residual 范围，同时加入候选间 coverage / diversity loss。

---

### `L_score`

只负责：

> **在已经产生的候选中选哪一条。**

不要让 score 去弥补 candidate generation 本身的能力不足。

---

# 十六、训练顺序应该吸取 V4.8.3 的经验

你之前 V4.8.3 candidate-only 训练一个非常有价值的结果就是：在固定 Pillar 闭环里，可行候选平均数从 5.49 增加到 7.35，zero-feasible 从 5.82% 降到 0.88%，同时仍保持零碰撞。

这说明你的 DE-P 迭代里已经实际证明：

> **先把候选“造好”，往往比先把 score“排好”更重要。**

所以 DE-P-Car 也建议明确两阶段。

---

# 十七、第一阶段：Candidate Capacity Training

第一阶段暂时不要把主要精力放在 score。

训练：

```text
Depth Encoder
LiDAR Encoder
Fusion
Candidate Head
```

重点优化：

```text
static safety
path guidance
kinodynamic
comfort
candidate diversity
```

Score 可以：

```text
冻结
```

或者只给很小权重。

这一阶段最重要的 validation metric 应该是：

```text
Best-of-15 feasible rate
Average feasible candidates/frame
Zero-feasible frame rate
Best candidate clearance
Best candidate path progress
Kinematic violation rate
```

而不是：

```text
network top-1 accuracy
```

---

# 十八、第二阶段：Score Calibration

只有第一阶段满足：

```text
15 candidates本身有足够安全通道
```

之后才：

```text
freeze / partially freeze
sensor encoder + candidate geometry
```

再训练 score head。

而且我建议这次一开始就把：

```text
candidate head
score head
```

真正独立。

你现在的 `DepHead` 已经支持 `unified / split / independent` 三种 variant，并可以分别获取 candidate 和 score 参数。

DE-P-Car 可以直接吸收这个经验：

```text
Perception/Fusion trunk
      ├── Candidate Tower
      └── Score Tower
```

我会推荐从第一版就用：

```text
independent
```

或者至少：

```text
split
```

不要重新回到 unified head。

---

# 十九、动态障碍先不要塞进新网络训练

这一点我建议继续保持 Route-A 的思路。

你现在的 V4.9.1 静态网络训练代码其实明确把 dynamic module 和 dynamic loss 关闭了。`MixedSceneStaticYOPOV1` 会建立一个 dynamic-disabled 的 `DepNetwork`，训练 objective 也明确关闭 dynamic loss。

Car 版本第一轮也建议：

```text
Learned Network:
static geometry + local planning
```

动态：

```text
LiDAR
↓
clustering
↓
tracking
↓
Kalman prediction
↓
bounded reachability
↓
runtime candidate veto/ranking
```

继续放在网络外。

这能显著降低你重新训练 Car 网络的难度。

---

# 二十、不过 LiDAR 网络输入和动态 LiDAR tracking 可以共用同一个传感器

最终会是：

```text
                         VLP-16
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        LiDAR BEV                  Dynamic frontend
             │                           │
             ▼                           ▼
      Learned planner              TrackManager
             │                           │
             │                       Kalman
             │                           │
             │                  Reachability tube
             │                           │
             └─────────────┬─────────────┘
                           ▼
                   Runtime Safety
```

两边使用同一个 LiDAR。

但：

```text
Learned LiDAR branch
```

主要学习**空间几何**；

```text
Dynamic branch
```

负责**时间运动信息**。

职责非常清晰。

---

# 二十一、你的数据生成也应该因此重新设计

我建议一个训练 sample 不再只是：

```text
depth + state
```

而是：

```text
sample/
│
├── sensor
│   ├── depth.npy
│   ├── lidar.npy
│   ├── depth_timestamp
│   └── lidar_timestamp
│
├── vehicle
│   ├── pose
│   ├── velocity
│   ├── acceleration
│   ├── steering_angle
│   └── yaw_rate
│
├── route
│   ├── mission_goal
│   ├── local_subgoal
│   ├── local_path
│   └── reference_curvature
│
└── authority
    ├── map_uuid
    ├── occupancy
    └── map_hash
```

然后 loader 实时/预处理产生：

```text
depth_tensor
lidar_bev
state_9d
path_reference
```

---

# 二十二、Depth 和 LiDAR 的同步必须在数据生成阶段就冻结

这次一定不要以后才补。

一个 sample 必须明确绑定：

```text
depth timestamp
lidar timestamp
odom timestamp
steering timestamp
TF timestamp
```

必须设置最大允许 skew。

超出就：

```text
sample invalid
```

而不是：

```text
取最近一帧凑合
```

特别是小车运动时，LiDAR scan 与 camera frame 不同步会让：

```text
前方障碍
```

在两个传感器里出现不同位置。

模型会被迫学习传感器错位。

---

# 二十三、建议额外加入 modality validity

网络输入再多几个 flag：

```text
depth_valid
lidar_valid
```

训练时可以适当做：

```text
Depth dropout
LiDAR beam dropout
sensor noise
```

但禁止两个传感器同时失效。

这样未来实际传感器某一帧掉包，系统不会完全崩。

不过正式 runtime：

```text
sensor validity
```

仍应该由安全层处理，不能只相信网络鲁棒性。

---

# 二十四、数据集第一轮不要太大

你之前 DE-P 数据链已经吃过“大规模生成以后才发现 contract 有问题”的亏。

所以这里建议严格：

```text
Pilot
↓
Small training
↓
Closed loop
↓
Formal dataset
```

例如先做：

```text
20~50张随机地图
```

再生成一个小型：

```text
10k~30k samples
```

重点验证：

```text
depth/lidar同步
TF
state
Hybrid A* local path
15 candidate rollout
footprint safety
loss gradient
loader
training convergence
closed-loop
```

全部正确以后，再生成正式数据。

---

# 二十五、地图 split 一定按 Map UUID

不要随机拆 frame：

```text
random.sample(frames)
```

而应该：

```text
train maps
validation maps
test maps
```

完全隔离。

否则同一张随机地图：

```text
不同位置
```

同时进入 train 和 val，会让 validation 过于乐观。

这个原则继续沿用你当前 DE-P 已经形成的 manifest/hash/split contract 会很好。

---

# 二十六、我建议的完整训练路线

整体可以压缩成这条链：

```text
P1
Ackermann trajectory contract
3×5 speed-curvature lattice
bicycle rollout
vehicle footprint
        │
        ▼
P2
multimodal data contract
depth + LiDAR + state + local path
        │
        ▼
P3
10k~30k pilot dataset
        │
        ▼
P4
Depth-only Car baseline
验证trajectory/loss是否正确
        │
        ▼
P5
LiDAR-only baseline
        │
        ▼
P6
Depth + LiDAR fusion
Candidate Capacity Training
        │
        ▼
P7
Candidate gate PASS
        │
        ▼
P8
Score calibration
        │
        ▼
P9
static closed-loop
random unseen maps
        │
        ▼
P10
接入V4.9.1 deterministic
dynamic safety
        │
        ▼
P11
Ackermann recovery
```

这里有一个很关键的实验顺序：

```text
Depth only
LiDAR only
Depth + LiDAR
```

一定三个都跑。

否则最后融合网络表现提高了，你无法证明：

> 是多模态真的有效，还是只是 LiDAR 一路在起作用。

这对于之后写论文也很重要。

---

# 二十七、最终网络我建议收敛成这样

```text
DEPCarNetV1

Depth:
    [1,96,160]
        ↓
    MobileNetV3-Small
        ↓
    Depth feature

LiDAR:
    VLP16 point cloud
        ↓
    base_link transform
        ↓
    BEV representation
        ↓
    Lightweight CNN
        ↓
    BEV feature map

Route/vehicle:
    [v,a,δ,yaw_rate,
     gx,gy,
     sin(eψ),cos(eψ),
     κref]
        ↓
    State MLP

Lattice:
    SpeedEmbedding[3]
    CurvatureEmbedding[5]
        ↓
    15 Candidate Queries

Each Candidate:
    Depth directional feature
    +
    BEV trajectory-sampled feature
    +
    State feature
    +
    Candidate embedding
        ↓
    Candidate MLP
        ↓
    Δδ_mid
    Δδ_end
    Δv_end
    ΔT

Separate Score Tower:
        ↓
    score

Candidate parameters
        ↓
AckermannRolloutV1
        ↓
15 future vehicle trajectories
        ↓
training loss / runtime safety
```

这会是我目前最推荐的 **DE-P-Car 第一代正式网络**。

---

# 二十八、其中哪些 V4.9.1 代码可以直接借，哪些要重写

| V4.9.1                          | DE-P-Car                  |
| ------------------------------- | ------------------------- |
| `DepBackbone`                   | 保留思路，Depth branch 可初始化权重  |
| MobileNetV3                     | **可以保留**                  |
| `observation_dim=9`             | 可以保持维数，但重定义语义             |
| `StateTransform`                | **重写**                    |
| 3×5 lattice                     | **保留 15 candidate 思想**    |
| pitch×yaw lattice               | 改 speed×curvature         |
| `DepHead`                       | 借鉴 split/independent，主体重写 |
| 9D PVA output                   | **删除**                    |
| `pred_to_endstate()`            | 改 `AckermannRolloutV1`    |
| candidate score                 | **保留**                    |
| dynamic attention               | 第一阶段不进网络                  |
| static differentiable objective | **保留训练哲学**                |
| vehicle sphere safety           | 改 swept footprint         |
| V4.8.3 candidate-first          | **强烈建议继承**                |

---

## 最核心的一句话

你现在不要把任务理解成：

> **“用深度图和 LiDAR 训练一个网络直接预测小车方向盘。”**

而应该理解成：

> **“用深度图提供前向高密度几何、用 360° LiDAR 提供地面平面空间结构，再让神经网络针对 15 个 Ackermann 运动原语生成可微调的未来轨迹；训练时通过完整车辆 footprint、Hybrid A* 局部引导和运动学约束直接优化这些未来轨迹，最后再单独学习候选排序。”**

这既保留了你 V4.9.1 DE-P 最有价值的 **“You Only Plan Once / multi-candidate trajectory prediction”** 框架，又真正把网络的物理基础从无人机改成了 Ackermann 小车，而不是简单删除 `z` 轴。
