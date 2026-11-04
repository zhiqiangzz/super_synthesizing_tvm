# 数值实验开发交接：MPFR 检验与实际参考状态对照

## 工作边界与目标

请在开发 worktree 中实现、测试和运行以下任务。论文 worktree
`/home/zhiqiangz/workspace/tvm/.claude/worktrees/paper_writing` 只负责写作、引用、结果导入和排版，
不得在其中修改编译器、新增数值测试或运行实验采集。

已核对的开发基线为 `main@d2b0653238e35d4698fdb2af27200e2f66270852`，包含
`[FEAT][TE] Validate fusion accuracy with MPFR and Hypothesis`。
论文分支已 rebase，HEAD 为 `ae253675232f7d5a6a9f02c62e3343db2c79af48`。
如果开发基线继续变化，请记录最终版本并重新核对实验接口。

交付两部分证据：

1. 按现行 MPFR 方法重新测量论文已有精度实验及受验收影响的融合结果和耗时。
2. 对方差和 softmax 比较源程序、实际参考状态 B、维护状态三种实现，检验状态表示的数值影响。

保持状态选择算法、精度验收阈值和预先指定的扫描范围不变。实验允许发现维护状态更差的情况。
若发现实现错误，请单独报告和修复，并记录影响；不要为了得到预期优势调整算法或筛选样本。

## 独立实验入口

建议新增 `super_opt/examples/audit_numerical_accuracy.py`，通过 `--output-dir` 导出结果。
复用 `chain_ops`、现有 LLVM 编译路径、MPFR 求值和误差统计接口。
入口不得依赖论文目录。论文特有的异常输入和先前工作对照程序可从
`Hapax/scripts/` 只读迁移；必要的实验定义应在开发侧自包含。

### 实际参考状态 B 的程序生成

已只读验证以下流程可以为方差和 softmax 构造 B：

1. 取 registry 中该算子的原 TE 输入与输出，发现唯一归约链。
2. 构造 `_Select(_Chain(chain, members_of(chain), TeSigns(outputs)))` 并运行一次状态选择。
3. 通过 `selector.base()` 取得经过定义域修复的初始状态。按照
   `reducer/states.py::_Select.supplement` 的构造，收集这些状态及所需输出定义中的
   `atoms_in(..., R)`，得到实际的 B；不要使用 `judge()` 返回的 Q 的字符串。
4. 使用 `derive(B_atoms, required_full, chain.axis, preserve=True)`。
5. 分别通过 `build_chain` 和独立的 `Rewriter` 生成 B 程序和同次选择所得的维护状态程序。

必须保留参考元组的全部分量及生成的结果重建表达式：

| 算子 | B 的分量 | 重建方式 |
| --- | --- | --- |
| 方差 | sum(x), sum(x²), count | ss = sum(x²) - sum(x)²/m，再保留原输出 var = ss/m |
| softmax | max(x), sum(exp(x)) | den = sum(exp(x)) * exp(-max(x))，再保留原输出 exp(x-max(x))/den |

`preserve=True` 确保方差的 count 不因最终表达式使用全轴长度 m 而被删去。
softmax 不能手工约成 `exp(x)/sum(exp(x))`，这会改变浮点实验对象。
参考状态版本必须直接执行；将它先送入 AccuracyGate 会因无界指数等静态条件被拒绝，导致未测量。
静态检查诊断可另行记录。

### 对齐执行条件

三种程序共享完全相同的已量化 float32 数组，使用同一 LLVM 配置和递增索引的串行归约。
关闭 fast-math、重结合和收缩；记录完整 target 和生成代码。
两个元组归约器沿用现有首元素初始化。

MPFR 共同参考由原 TE 在这些已量化输入上求值得到。
使用 `high_precision_references()` 和 `numeric.errors()`，在计算误差之前不要转为 float64。
MPFR 精度从 128 位开始加倍，最高 2048 位；采用现有一致性判定，不另写宽松的判断。

## 预先固定的代表性实验

- 形状 `(3,m)`，m 为 16、256、4096；随机种子 0、1、2。
- 基础数据从整数 [-16,16] 除以 8 得到，每行强制包含 -2、0、2，再按种子打乱。
  对同一尺寸和种子复用基础数据，三种实现始终使用相同的实际 float32 数组。
- 方差偏移扫描（尺度 1）：0、1、16、256、4096、10000、65536、1048576。
  此范围保留 1/8 网格，避免将输入离散程度因量化消失误认为算法消去。
- softmax 偏移扫描（尺度 1）：-128、-112、-104、-96、-88、-64、0、64、80、88、96、128、10000。
- 方差尺度扫描（偏移 0）：2 的 -80、-70、-60、-30、0、30、60、65、70 次方。
- softmax 尺度扫描（偏移 0）：1/16、1、4、16、32、64、128。
- 单独测试单元素、全零和常量输入。

保留源程序失败、维护状态更差、NaN/Inf、域外输入和 MPFR 未收敛等结果。
有限误差与非有限结果数量分别报告；零参考结果使用绝对误差并标注是否精确相等。
不要把失败样本从分母中删除。

## 已有实验的更新

- 重新测量 registry 中源程序、手工融合和综合版本的 centred/offset 比较；每个 case 只生成一组
  输入，并对原 TE 求一次 MPFR 参考，三种实现共享该结果。
- 导出完整 `dataclasses.asdict(AccuracyReport)`，保存每输出名称、dtype、误差指标、实际 extents、
  `sensitivity`、`reference_precision` 和有效 `sensitivity_samples`。
- 保留实际 AccuracyConfig。默认逐元素判定为：
  `abs(candidate-reference) <= 100*max(abs(source-reference), sampled_sensitivity) + 64*eps(dtype)*abs(reference)`，
  绝对容差默认零。敏感性为相邻浮点数扰动的输出变化，不是条件数上界。
- 用实际采样尺寸，而非无条件假设归约长度 256、其他长度 3；现行逐操作迭代体积上限为 16384。
- 对异常输入及先前工作对照重新运行相关数值检查，并明确区分没有执行、被拒绝和已通过。
- 重新测量完整融合耗时：预热一次，之后五次运行取中位数。单独记录推导/证明与验收的阶段口径，
  不把旧的 float64 验收耗时作为新版结果。
- 不从 normwise 汇总误差推导逐元素验收的阈值利用率；现有报告没有导出足够信息。

当前论文采集器仍读取已删除的 `FamilyError.inherent`，且会从旧标量误差计算阈值比例。
不要运行这些旧采集器并把结果重标为新方法。

## 交付格式

输出独立结果目录：

| 文件 | 内容 |
| --- | --- |
| manifest.json | 代码提交、工作树状态、源码和脚本指纹、依赖版本、LLVM 参数、完整配置和运行命令 |
| measurements.jsonl | 每个 case/实现/输出的误差指标、参考精度、实际尺寸、状态和失败原因 |
| accuracy_reports.json | 新版完整验收报告及配置 |
| summary.csv | 可制表作图的长表；明确区分有限误差、精确零、非有限值和未执行 |
| inputs.npz | 实际量化输入，按 case ID 和内容哈希关联 |
| programs/ | 源程序、B 程序、维护状态程序及状态定义、结果重建和证明结果 |
| pytest.log、README.md | 测试结果、字段定义及一条重现命令 |

每个误差记录包含 absolute、relative、normwise、ulps、nonfinite；ULP 是诊断量，不是统一验收限制。
源误差为零时，区分“两者都为零”与“候选误差非零”；不要把数值零用 truthiness 转成缺失值。
源数据文件中明确编码非有限状态，不依赖 JSON 消费者接受裸 NaN/Infinity。
记录 gmpy2、MPFR、Hypothesis 及编译依赖的实际版本。

## 测试与完成标准

建议新增 `tests/python/te/superopt/test_reference_state_accuracy.py`，检查：

- B 的实际构造、全部分量保留和结果重建；代数验证确立与源程序的实数语义关系。
- 三方输入、编译参数和归约次序一致；B 不被静态 gate 提前过滤。
- 单元素、常量、零结果、非有限结果和 MPFR 未收敛可被完整记录。
- 报告使用新字段和真实逐元素验收结果；不使用旧阈值公式。
- 运行现有 numeric、accuracy、Hypothesis 及完整 superopt 测试。

不要断言维护状态必须在每个输入上更准确。验收标准是实验对象正确、记录完整、可从提交和命令复现。
交付代码提交、结果包位置、测试记录及异常案例说明。

## 论文侧接收

论文 session 只导入结果并更新文字、引用和排版。新精度统计、实际 B 对照及包含新验收开销的耗时，
必须在收到结果包后更新。旧 snapshot 保留来源，不充当新方法的测量结果。
现有手写方差/softmax 例子仅说明机制，不等同于此次 LLVM 三方对照。
