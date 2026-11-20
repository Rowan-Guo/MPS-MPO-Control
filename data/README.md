# 实测源荷数据及引用 / Provenance

## 馈线拓扑与参数

33 节点网络使用 Baran 和 Wu 的馈线重构算例。三种接线共用 33 组节点负荷及相同的物理线路参数。常态接线包含 32 条闭合线路，全部 5 条联络线断开。两种重构工况分别闭合 8–21、断开 7–8，以及闭合 18–33、断开 13–14。这里选择的开关状态用于验证控制算法，每个工况内固定。逐项核对 MATPOWER 后，将历史代码中 14–15 的电阻从 0.5916 Ω 订正为 0.5910 Ω。[算例数据](https://github.com/MATPOWER/matpower/blob/b6c85b0323d20fb801cab3377c803a285d730847/data/case33bw.m) · [Baran 与 Wu，1989](https://doi.org/10.1109/61.25627)

`case69.csv` 转录了 Baran 和 Wu 的 69 节点径向馈线经 MATPOWER 整理的数值参数。每行包含节点编号、入线的上游节点、线路电阻和电抗（Ω）、节点有功和无功负荷（kW、kvar）。1 号节点为平衡节点，上游编号和入线阻抗用零表示。[算例数据](https://github.com/MATPOWER/matpower/blob/b6c85b0323d20fb801cab3377c803a285d730847/data/case69.m) · [Baran 与 Wu，1989](https://doi.org/10.1109/61.19265) · [MATPOWER 算例数据声明](https://github.com/MATPOWER/matpower/blob/b6c85b0323d20fb801cab3377c803a285d730847/LICENSE)

两套算例的基准功率均为 10 MVA，基准线电压均为 12.66 kV。节点功率的原始数值除以 10,000，线路阻抗除以 12.66²/10，得到各自的标幺值。将实验负荷比例与线路倍率设为 1 可恢复原始负荷及阻抗。69 节点算例的线路倍率作用于 1–2–…–9–53–54–…–65 路径上的线路。33 节点算例的倍率作用于原 1–2–…–18 主干上的在运线路。

## 半小时实测源荷

`ausgrid_customer12_2011_2012.csv` 来自 Ausgrid Solar Home Electricity Data 中匿名住宅 **Customer 12** 的 2011-07-01 至 2012-06-30 半小时电力测量序列，经 Pierre Haessig 的公开 `solarhome-control-bench` 项目整理，保留 `GC`（一般用电）和 `GG`（光伏发电）两列，单位为 **kW**。共有 17,568 条、30 分钟连续采样。内容没有个人身份信息。

- 原始数据发布者为 Ausgrid。Pierre Haessig 的归档说明将原始数据授权列为 [CC BY 3.0 Australia](https://creativecommons.org/licenses/by/3.0/au/)。
- 原始测量背景：Ratnam, Weller, Kellett & Murray, *Residential load and rooftop PV generation: an Australian distribution network dataset*, International Journal of Sustainable Energy, DOI: [10.1080/14786451.2015.1100196](https://doi.org/10.1080/14786451.2015.1100196).
- 开放数据及本文件直接来源：[pierre-haessig/solarhome-control-bench/data/data_2011-2012.csv](https://github.com/pierre-haessig/solarhome-control-bench/blob/master/data/data_2011-2012.csv)；[数据说明](https://github.com/pierre-haessig/solarhome-control-bench/blob/master/data/README.md)。
- Pierre Haessig 的 2011–2012 年度整理版采用 [CC BY 4.0](https://github.com/pierre-haessig/ausgrid-solar-data/blob/5bdbc2d4edd73e137667cdadd51ca6b6059fedbe/LICENSE)。本项目基于该版本构造节点注入并进行标幺缩放。

**空间真实性边界：** 这是 **1 户** 的真实源荷序列，不是 IEEE 33 节点的现场测量。`measured_profiles.py` 将不同日期、相同时段的成对 GC/GG 曲线映射到 12 个随机节点，并按标准馈线原始负荷、光伏容量比例缩放。因此本实验是“**真实拓扑参数 + 真实时间曲线驱动的合成节点注入**”组合，而不是某条实际运营配电网的数字孪生。保留此说明便于准确解释相关性与泛化结果。

训练 / 测试窗口分离：训练输出 2011-07-01 起 160 天；测试输出从第 215 天起 115 天，两个窗口之间另留出空间映射最大 33 天滞后，避免未来原始读数泄漏到训练中。四个时间段的 EM 模型只使用训练数据。运行时用实时可测节点的三档离散状态和时段标签选择动作。

## 下一计量年度的独立回放

`ausgrid_customer12_2012_2013_wh.csv` 源自 Ausgrid 的 2012-07-01 至 2013-06-30 原始计量年度，由 Pierre Haessig 归档。我们从归档中的 `Solar home 2012-2013.csv` 提取 Customer 12 的 GC 和 GG 两个通道，共 365 天、730 行，所有记录均为原始表计实测质量标记（Row Quality 为空）。保留日期、通道和一天 48 个半小时的电量，数值由 kWh 精确换算为整数 Wh。原始年度 CSV 的 SHA-256 为 `e4be46b3c9991c65735a9545319a2f1c33be8e639c75a17a88e0323a8e5098be`。[数据归档](https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip) · [归档说明](https://github.com/pierre-haessig/ausgrid-solar-data/blob/5bdbc2d4edd73e137667cdadd51ca6b6059fedbe/README.md) · [CC BY 3.0 Australia](https://creativecommons.org/licenses/by/3.0/au/)

列标题表示区间结束时刻，0:30 列对应当天 00:00–00:30。读入时将 Wh 除以 500 得到半小时平均 kW，以每个区间的起点标记时间。用同样的转换读取原始归档中的 2011–2012 年度，可在浮点误差内复现既有训练文件的全部读数。

2011–2012 年度用于算法开发和同年度的分窗对照。确定候选策略和评分规则后，再用 2012–2013 年度检验同一住户的跨年表现。原训练年度确定的负荷与光伏归一化尺度、分类阈值和策略参数全部固定。33 天的最大节点偏移使每个场景保留 332 天、15,936 个半小时。
