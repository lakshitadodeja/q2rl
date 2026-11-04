# Flow Matching BC：基于官方 Diffusion Policy 的修改版

本版替换了先前的精简实现。保留官方算法文件的方法顺序、函数说明、观察编码、动作分块、EMA、训练状态保存和归一化层替换工具。
代码尚未用真实 Lift 数据训练，不能据此声称任何成功率。

## 来源和可核对差异

- 上游：ARISE-Initiative/robomimic 的 `robomimic/algo/diffusion_policy.py`，具体 commit 记录在代码文件头。
- `models/flow_policy_nets.py` 复用同一来源的 ConditionalUnet1D。唯一实现修改是 Python 标量时间采用浮点 dtype，避免连续时间被截断；保留原许可。
- `models/flow_ema.py` 提取自 diffusers **v0.11.1** 的原始 EMAModel 类，未改变其算法、更新公式或 warmup；附带 Apache-2.0 许可。这样无需把旧版 diffusers 和它的旧 huggingface-hub 依赖装入现有环境。
- `docs/flow_policy_vs_diffusion.patch` 提供算法文件相对上游的完整差异；代码中 `FLOW CHANGE` 是算法替换，`COMPATIBILITY` 是旧项目接口适配。

## Flow 必需的变化

1. 删除 DDPM/DDIM scheduler。
2. 训练样本改为 `x_t = (1-t)*noise + t*actions`，训练目标改为 `actions-noise`。
3. 用 velocity 网络预测变化方向，仍使用 MSE。
4. 推理用 Euler 积分，从 t=0 的噪声推进到 t=1 的动作。
5. 算法注册名、配置和网络输出名称更新为 flow/velocity。

`time_scale=1000` 用于时间嵌入，积分区间仍为 [0,1]；默认采样 10 步。这些是 Flow 参数选择，不是 OGPO 配置的逐项复现。默认保留动作范围裁剪。

## 保留和恢复的功能

- 单帧或多帧观察：默认 2 帧，frame_stack 必须与 observation_horizon 一致。
- 默认预测 16 个动作，从索引 To-1 开始执行 8 个动作，然后重新生成。
- 官方 U-Net 默认宽度 [256,512,1024]、时间嵌入 256。
- 官方 EMA 的 power=0.75，允许关闭 EMA。
- 保存并可恢复优化器、学习率调度器、模型权重和 EMA 权重；额外保存 EMA 更新计数。
- 保留 BatchNorm->GroupNorm 工具。当前交付验证范围是低维 Lift，不声称已经验证图像训练。

## 必要接口适配与剩余差异

q2rl 的旧 RolloutPolicy 每次只传当前一帧，因此增加历史队列，并在 episode 开始时重复首帧填充。对已经堆叠好的观察也支持直接输入，避免重复堆叠。训练数据与预测动作从 To-1 开始执行的对应关系有测试。

配置通过旧项目的 BCConfig 复用优化器配置字段；使用 AdamW、学习率 1e-4、L2=1e-6。旧版训练基类不支持上游新版逐 batch cosine scheduler，本版显式使用旧框架的恒定学习率配置，**不是上游所有训练超参数完全相同**。

恢复接口为 `deserialize(checkpoint, load_optimizers=True)`。现有 train.py 没有因此自动新增命令行断点续训功能；epoch、数据迭代器和随机数状态也未保存，不能声称整场训练可逐位复现地续跑。
旧的 149 行试验版 checkpoint 与本版结构不同，不支持直接加载。

## 测试（项目根目录）

```bash
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
python robomimic/tests/test_flow_policy.py
```

使用小型网络进行 CPU 测试：路径方向、训练和验证、模型重载、RolloutPolicy、历史队列、单帧/关闭 EMA、恢复优化器与 EMA 后同一随机数下的下一步更新，以及合成 HDF5 经真实 SequenceDataset 的帧/动作对齐。

## Oscar 上训练（数据准备好后）

在已有可用的 GPU 作业和仿真环境内，确认使用本分支的 robomimic：

```bash
export PYTHONPATH="$PWD/robomimic${PYTHONPATH:+:$PYTHONPATH}"
python -c "import robomimic; print(robomimic.__file__)"
python robomimic/robomimic/scripts/train.py \
  --config experiments/configs/flow/lift_low_dim.json \
  --dataset /absolute/path/to/lift-low-dimensional.hdf5 \
  --debug
```

数据路径是占位符，需替换真实 Lift 数据。先统一双方的数据版本、观察历史和动作长度，debug 集成检查通过后再去掉 --debug。JSON 是初步试跑配置：100 epoch，每 epoch 100 次更新，并非已经调优的训练方案。
训练后的 checkpoint 使用现有 run_trained_agent.py 评估。

修改只在本地 Flow-Matching 工作目录中，尚未 commit / push。数据、日志和模型不加入 Git 提交。
