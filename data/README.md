# 实测源荷数据及引用 / Provenance

`ausgrid_customer12_2011_2012.csv` 来自 Ausgrid Solar Home Electricity Data 中匿名住宅 **Customer 12** 的 2011-07-01 至 2012-06-30 半小时电力测量序列，经 Pierre Haessig 的公开 `solarhome-control-bench` 项目整理，保留 `GC`（一般用电）和 `GG`（光伏发电）两列，单位为 **kW**。共有 17,568 条、30 分钟连续采样。内容没有个人身份信息。

- 原始数据发布者：Ausgrid，Solar Home Electricity Data（CC BY 4.0）。
- 原始测量背景：Ratnam, Weller, Kellett & Murray, *Residential load and rooftop PV generation: an Australian distribution network dataset*, International Journal of Sustainable Energy, DOI: [10.1080/14786451.2015.1100196](https://doi.org/10.1080/14786451.2015.1100196).
- 开放数据及本文件直接来源：[pierre-haessig/solarhome-control-bench/data/data_2011-2012.csv](https://github.com/pierre-haessig/solarhome-control-bench/blob/master/data/data_2011-2012.csv)；[数据说明](https://github.com/pierre-haessig/solarhome-control-bench/blob/master/data/README.md)。
- License：Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))，保留上述署名并注明此项目对它所进行的空间映射和标幺缩放。

**空间真实性边界：** 这是 **1 户** 的真实源荷序列，不是 IEEE 33 节点的现场测量。`measured_profiles.py` 将不同日期、相同时段的成对 GC/GG 曲线映射到 12 个随机节点，并按标准馈线原始负荷、光伏容量比例缩放。因此本实验是“**真实拓扑参数 + 真实时间曲线驱动的合成节点注入**”组合，而不是某条实际运营配电网的数字孪生。保留此说明便于准确解释相关性与泛化结果。

训练 / 测试窗口分离：训练输出 2011-07-01 起 160 天；测试输出从第 215 天起 115 天，两个窗口之间另留出空间映射最大 33 天滞后，避免未来原始读数泄漏到训练中。四个时间段的 EM 模型只使用训练数据。运行时用实时可测节点的三档离散状态和时段标签选择动作。
