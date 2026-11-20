# MPS–MPO Control

### 面向部分可观测配电网的低秩概率电压控制
**Low-Rank Probabilistic Voltage Control for Partially Observed Distribution Networks**

本项目将**负荷联合概率的 MPS/TT 表示**与**物理驱动的控制损失 MPO 表示**结合，通过张量缩并计算观测条件下的动作期望损失，实现考虑多节点相关不确定性的电压控制决策。

项目从 10 节点精确交流潮流验证出发，进一步实现了**高维概率学习、非枚举损失算子构造、张量秩控制和独立 AC 潮流性能评价**，形成可复现的研究框架。

## 研究成果一览

| 关键成果 | 已实现结果 |
|---|---|
| **高维联合状态空间** | 16 个三档随机负荷，**43,046,721** 种可能状态 |
| **低秩 MPO 存储** | 单动作 **45,936** 个标量，相比完整损失表约 **937 倍缩减** |
| **高维算子构建** | 16 变量、6 个控制动作，仅需 **198 次离线 AC 校准潮流** |
| **控制效果** | 10 变量合成算例中，独立 AC 样本上的平均电压损失比条件均值法**降低约 8.1%** |
| **数据驱动概率表示** | EM 从 **6,000 条训练样本**学习相关概率，并构造非负概率 TT |
| **正确性与可复现性** | **14 项**单元及回归测试，提供固定种子实验 CSV |

以上指标对应仓库中 `examples/scaling_known.csv` 和 `examples/scaling_learned.csv` 的可复现实验配置。

## 1. 核心方法

### 1.1 MPS/TT：紧凑表达相关联合概率

面对多个缺测负荷，决策需要在完整**条件联合分布**上评估控制动作，而非仅对平均负荷做一次潮流计算。

项目支持两条概率建模路径：

- **Born-MPS：** 将已知概率表的平方根写成张量网络，基于双层期望缩并完成小规模精确验证，并提供已知低秩高维概率的对照。
- **样本学习概率 TT：** 使用 EM 拟合有限混合分布，直接得到非负、归一化的概率张量网络，实现从负荷样本到条件概率控制的端到端流程。

已知概率的 Born 表示满足

\[
\psi(x)=\sqrt{p(x)},\qquad
J_a(y)=\frac{\langle\psi|P_y O_a P_y|\psi\rangle}
{\langle\psi|P_y|\psi\rangle}.
\]

样本学习的概率 TT 则直接表示

\[
p(x_1,\ldots,x_m)=\sum_{k=1}^{C} w_k\prod_{i=1}^{m}p_{ki}(x_i).
\]

其中 `C` 为混合分量数，概率 TT 的键维不超过 `C`，在线使用单层概率缩并。每个三档负荷对应**一个局部物理维度为 3 的张量站点**，避免二进制编码的冗余状态。

### 1.2 MPO：由交流潮流灵敏度直接构造高维损失算子

对每个控制动作 `a`，使用参考运行点及有限差分 AC 潮流校准末端电压的仿射近似：

\[
\Delta V_a(x)=c_a+\sum_{i=1}^{m}f_{a,i}(x_i).
\]

在此模型上，对电压死区损失

\[
L_a(x)=\max(0,|\Delta V_a(x)|-d)
\]

进行余弦 Fourier 展开，利用三角函数加法结构**直接生成对角 MPO 核**。在采用 `K` 个谐波时，中间键维至多为

\[
r_{\mathrm{MPO}}\le 2K+1.
\]

MPO 构建不需要生成 `3^m` 个状态的损失数组。六个动作的 AC 校准次数为

\[
N_{\mathrm{AC}}=6(2m+1).
\]

项目还实现了一阶 LinDistFlow 灵敏度构建选项，可用于比较纯解析与 AC 校准两条物理建模路径。

### 1.3 决策：条件期望缩并与误差诊断

给定观测 `y`，对于每个动作计算

\[
J_a(y)=\mathbb E[L_a(X)\mid Y=y],
\qquad
a^*(y)=\arg\min_{a\in\mathcal A}J_a(y).
\]

张量网络将高维条件求和转化为逐站点缩并；`max_bond` 与相对截断阈值用于控制表达规模。原始 10 节点流程还输出动作间隔、分数误差与决策后悔值，帮助评价**秩、精度与控制收益**之间的关系。

对于 Fourier 表示，在包含全部仿射电压偏差的区间 `[-D,D]` 上有

\[
\|L_{\mathrm{affine}}-L_{\mathrm{Fourier}}\|_\infty
\leq\frac{4D}{\pi^2 K}.
\]

若所有可行动作的评分误差不超过 `\varepsilon`，则近似选择的动作满足

\[
J_{\tilde a}-J_{a^*}\leq 2\varepsilon.
\]

该误差分析为进一步的**决策保持型张量秩选择**提供基础。

## 2. 高维实验结果

### 2.1 计算规模与表示效率

实验采用六个有功/无功控制动作、每个随机负荷三个离散状态以及 `K=16` 个 Fourier 谐波。

| 随机负荷数 `m` | 完整联合状态数 `3^m` | 单动作 MPO 标量数 | 六动作离线 AC 调用数 |
|---:|---:|---:|---:|
| 4 | 81 | 6,732 | 54 |
| 6 | 729 | 13,266 | 78 |
| 10 | 59,049 | 26,334 | 126 |
| **16** | **43,046,721** | **45,936** | **198** |

随着变量维数增加，完整状态表呈指数增长；而所构造的 Fourier MPO 规模随站点数线性增加。在较小规模（`m=4,6`）时，直接表仍更紧凑；高维场景则显示出明显的张量表示收益。

### 2.2 独立交流潮流控制效果

在**AC 校准替代模型 + 已知联合概率**的配置下，使用另外抽取的 400 个条件样本执行非线性 AC 潮流评价：

| 随机负荷数 | 条件均值动作平均损失 | MPS–MPO 动作平均损失 | 平均改善 |
|---:|---:|---:|---:|
| 4 | 0.002088 | 0.002088 | 0 |
| 6 | 0.003843 | 0.003843 | 0 |
| **10** | **0.005848** | **0.005374** | **8.1%** |
| 16 | 0.008187 | 0.007919 | 3.3% |

损失单位为 p.u.。`m=10` 时，平均损失差约为 `4.74×10⁻⁴` p.u.，成对正态近似 95% 区间为 `[2.18×10⁻⁴, 7.30×10⁻⁴]` p.u.。`m=16` 的单次结果同样为正，95% 区间为 `[-1.42×10⁻⁴, 6.77×10⁻⁴]` p.u.，显示了进一步进行多种子评价的研究价值。

### 2.3 从样本到概率决策

`probability_tt.py` 已实现 **EM → 有限混合联合概率 → 非负概率 TT → 条件缩并 → 动作选择**的完整流程。

- 三分量 EM 模型，6,000 条合成训练样本；
- 从训练样本直接得到低秩、可归一化的联合概率表示；
- 4、6、10、16 个随机负荷统一使用相同实验入口；
- 同时输出概率评分与已知分布参考的差异、在线缩并耗时、独立 AC 控制损失。

[已知概率实验结果](examples/scaling_known.csv) · [样本学习概率实验结果](examples/scaling_learned.csv)

## 3. 快速复现

**环境：** Python 3.10+、NumPy 1.24+。

```bash
python -m pip install -r requirements.txt
python -m unittest -v

# 原始 10 节点精确 AC 潮流实验
python run.py --out results/baseline

# 指定观测执行动作决策
python run.py --out results/baseline --decide B 20261120 0.05 0.10

# 键维限制与决策误差分析
python run.py --out results/rank2 --max-bond 2

# 已知相关概率：4/6/10/16 变量规模实验
python benchmark.py --sites 4 6 10 16 --harmonics 16 \
    --samples 10000 --ac-check 400 --out results/scaling_known.csv

# EM 概率学习：6000 条训练样本、三混合分量
python benchmark.py --sites 4 6 10 16 --harmonics 16 \
    --probability-source em-learned --train-samples 6000 \
    --components 3 --samples 10000 --ac-check 400 \
    --out results/scaling_learned.csv

# LinDistFlow 灵敏度对照
python benchmark.py --sites 4 6 10 16 --surrogate lin-distflow \
    --out results/scaling_lindistflow.csv
```

项目固定随机种子并保存实验 CSV，便于验证与扩展。

## 4. 项目结构

| 文件 | 作用 |
|---|---|
| `tensor.py` | TT-SVD、Born-MPS、对角 MPO、条件投影与缩并 |
| `probability_tt.py` | EM 联合概率学习、非负概率 TT、单层概率缩并 |
| `scalable.py` | 低秩相关 Born 分布、AC 灵敏度校准、Fourier MPO 解析构造 |
| `benchmark.py` | 高维规模实验、条件均值基线、独立 AC 评价 |
| `powerflow.py` | 原始 10 节点径向馈线交流潮流 |
| `run.py` | 10 节点完整实验、模型保存/重载与截断诊断 |
| `test_demo.py`、`test_scalable.py` | 14 项数值与回归测试 |
| `examples/` | 固定种子高维实验结果 |
| `config.json`、`requirements.txt` | 参数及依赖 |

## 5. 实验设置与研究拓展

当前代码包含两个互补实验层次：**10 节点算例**使用状态—动作级的非线性 AC 潮流损失和精确安全指示算子；**高维算例**使用合成径向馈线、AC 校准的仿射末端电压模型与 Fourier MPO，随后通过独立非线性 AC 潮流样本评价动作。上述 Fourier 误差界针对仿射损失表示；真实 AC 安全性及跨网络泛化可通过后续鲁棒校准和标准馈线实验进一步研究。

下一阶段的研究聚焦：**多工况与多随机种子验证、IEEE 标准配电网、非线性 AC 残差低秩修正、概率漂移适应、条件越限概率与风险约束控制**。

**方法基础：** [Tensor Train Decomposition — Oseledets (2011)](https://doi.org/10.1137/090752286)。

## IEEE 33 节点 + Ausgrid 实测源荷扩展（新）

在原始 10 节点和高维合成馈线实验之外，新增更接近工程场景的评测路径：

- **33 节点分支型 AC 潮流：** 采用 MATPOWER Baran–Wu IEEE 33 节点线路、原始基准负荷与 32 条常闭支路；双逆变器布置在节点 18、33。
- **实测源荷时序：** Ausgrid 2011-07 至 2012-06 匿名住宅 Customer 12 的 **17,568 条半小时真实负荷 GC / 光伏 GG 记录**，源文件和 CC BY 4.0 署名一并提交至 `data/`。
- **多点不完全观测：** 12 个随机源荷节点，2 个可观测，**3¹² = 531,441** 种离散状态；按时间段训练非负概率 TT，使用交流潮流校准的 Fourier MPO 条件缩并。
- **线性数量的校准：** 六种 P/Q 控制动作只需 **150 次离线 AC 潮流**获取两个监控节点的类别灵敏度。
- **公平的样本外测试：** 训练/测试原始读数严格按时间隔离；比较相关概率 TT、条件均值、独立概率 TT，使用完整 AC 潮流在未见过的连续负荷/PV 注入上评分，并采用按天分组的 bootstrap 区间。

**运行实验：**

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python realistic_benchmark.py --harmonics 16 --components 3 --max-test 320 \
  --out examples/ieee33_ausgrid_results.json
```

[详细方法与实验说明](docs/ieee33_ausgrid.md) · [数据来源与许可](data/README.md)

源荷曲线是实际测量；**节点级注入是将一户实测曲线通过分日期时移和容量缩放映射至 IEEE 33 节点的场景**，并非 33 个实测用户或真实运行中同一配电网的现场注入。

