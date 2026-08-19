# RoboNav DLA 部署

## 结论

`kinogoal_dla_resnet18_overfit_0819_epoch_500.pth` 可以部署到 Jetson Orin NX 的单颗 DLA 上，但不能把原始 PyTorch/ONNX 图原样交给 TensorRT。需要使用 [`tools/export_dla.py`](../tools/export_dla.py) 做等价图改写，再用 TensorRT 的 DLA-only 模式构建。

当前验证条件：Jetson Linux R36.4.3、TensorRT 10.3、FP16、batch=1、DLA Core 0、未启用 GPU fallback。最终 engine 构建成功，随机输入推理输出均为有限值。

## 网络输入和输出

脚本导出的是单 timestep 推理图。时序状态由 host 保存，并在每次调用时传入/取回：

| Tensor | Shape | 说明 |
|---|---|---|
| `rgb` | `[1, 3, 384, 512]` | 已完成图像预处理的 RGB |
| `pe` | `[1, 6, 192, 256]` | host 生成的 camera position embedding |
| `twist` | `[1, 3, 1, 1]` | 速度/角速度 |
| `delta` | `[1, 3, 1, 1]` | 已按 `[0.1, 0.1, 0.1]` 归一化 |
| `goal` | `[1, 6, 1, 1]` | 已按 `[10, 10, pi, 1, 1, 1]` 归一化 |
| `h`, `c` | `[1, 256, 1, 1]` | SRU 隐状态和 cell 状态 |

输出 trajectory 在 DLA 图中是 `[1, 7, 1, 20]`，host 侧转置为原模型使用的 `[1, 20, 7]`。DepthHead 的四个输出尺寸分别为 `[12,16]`、`[24,32]`、`[48,64]`、`[96,128]`。

相机渲染、PE 生成、RGB/深度预处理不在这个 engine 内。

## 原图中 DLA 不支持或不适合的部分

### `Hardsigmoid`

`FeatureModulation.spatial_gate` 使用 `nn.Hardsigmoid`。TensorRT 构建日志明确报告：

```text
Activation type: HARD_SIGMOID is not supported on DLA
```

改写为等价形式：

```text
Hardsigmoid(x) = ReLU6(x + 3) / 6
```

`+3` 和 `/6` 用固定权重的 1×1 Conv 表示，因此图中不再依赖 DLA 不支持的 Hardsigmoid 或常量层。

### `Linear/Gemm`

网络中的 FeatureModulation、SRU 和 TrajectoryHead 有多个 `Linear`。在本机 TensorRT/DLA-only 构建中，二维 `Gemm` 表达会导致 DLA 编译失败。脚本将它们改成保持 NCHW 的 1×1 Conv；权重只是从 `[out,in]` reshape 为 `[out,in,1,1]`，数学计算不变。

SRU 的 `linear_all` 进一步拆成四个门 Conv，避免导出后出现不适合 DLA 的二维门控图。

### SRU 中的同 tensor 自乘/自加

SRU 的门控公式包含 `f*f`、`a+a` 等同一个 tensor 作为两个输入的 ElementWise 节点。这类图在 DLA 编译器中无法稳定编译。脚本加入固定的 identity/negative identity 1×1 Conv，把两个输入显式分离；这些层没有可学习参数。

### 常量、动态 shape 和输出整理节点

原始 ONNX 图包含大量 `Constant`、动态 `Shape/Slice` 和 `Squeeze`。DLA-only 构建会在这些节点上失败，或产生 GPU reformat。脚本导出后使用 ONNX Runtime 的 `ORT_ENABLE_BASIC` 做静态图优化，并保持固定输入尺寸。

Trajectory 保持为 NCHW 输出，不在 engine 内执行最终 `permute/squeeze`；host 侧完成这一步。

### Resize

DLA 只支持预设 scale factor 的 Resize。当前网络的所有 Resize 尺寸是固定的 2 倍 nearest-neighbor，TensorRT 可以据此编译。不要把这些尺寸改成动态输入。

## 导出和构建

导出优化后的 ONNX：

```bash
python tools/export_dla.py \
  ckpts/kinogoal_dla_resnet18_overfit_0819_epoch_500.pth \
  /tmp/robonav_dla.onnx
```

当前这台机器的 `robonav_py312` 没有 PyTorch/ONNX/TensorRT；验证使用系统 Python。若 ONNX 只安装在临时路径，需要：

```bash
PYTHONPATH=/tmp/onnx-pkg/usr/lib/python3/dist-packages \
  /usr/bin/python3 tools/export_dla.py \
  ckpts/kinogoal_dla_resnet18_overfit_0819_epoch_500.pth \
  /tmp/robonav_dla.onnx
```

构建 DLA-only engine：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/robonav_dla.onnx \
  --saveEngine=/tmp/robonav_dla_core0.engine \
  --useDLACore=0 \
  --fp16 \
  --inputIOFormats=fp16:dla_linear \
  --outputIOFormats=fp16:dla_linear \
  --skipInference
```

这里故意没有加 `--allowGPUFallback`。构建日志必须包含：

```text
Allow GPU fallback for DLA: Disabled
PASSED
```

运行随机 dummy benchmark：

```bash
python tools/benchmark_dla.py /tmp/robonav_dla_core0.engine
```

脚本默认等价于 `trtexec --useDLACore=0 --warmUp=1000 --duration=10 --avgRuns=100`，输入由 TensorRT 自动生成 random dummy 数据。可用 `--duration`、`--warmup`、`--avg-runs` 和 `--export-times` 调整测试。

`DLA_LINEAR` 对宽度有对齐 stride。使用 TensorRT API 时应读取 binding stride；不要假设 `[1,C,1,1]` 的 raw buffer 只有 `C` 个 FP16 元素。`trtexec --loadInputs` 需要按实际 tensor stride 提供 padding。

## 本机实测

在 MAXN_SUPER、未锁定 `jetson_clocks` 的情况下，单 Core 0 结果为：

- 吞吐量：36.19 FPS
- 平均端到端延迟：28.14 ms
- P99 延迟：28.20 ms
- engine：43 MiB
- TensorRT execution context device memory：0 MiB
- trtexec 进程峰值 RSS：约 304 MiB
- Jetson 统一内存中的系统 RAM 增量：约 180 MiB，受缓存和后台进程影响

这是单颗 DLA Core 0 的结果，不是两个 DLA core 并行。FPS 只覆盖这个静态网络 engine，不包括相机输入和 PE 生成。
