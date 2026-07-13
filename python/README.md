# SymFT Python 接口

SymFT Python 接口是 C++ SymFT 模拟器的原生扩展封装，用于采样带噪声的
Stim 风格 Clifford+T 电路。主要能力包括：

- 从文本或 `.stim` 文件解析电路；
- 返回逐 shot 的测量记录或 detector 记录；
- 直接聚合 detector 丢弃数和逻辑错误数，避免保存大规模逐 shot 结果；
- 编译并复用 sampler，支持批处理和 counts 采样多线程；
- 通过 NumPy 返回普通布尔数组或位压缩数组。

当前包版本为 `0.1.0`，需要 Python 3.9+、NumPy 1.20+ 和支持 C++20 的编译器。

## 目录

- [安装与构建](#安装与构建)
- [快速开始](#快速开始)
- [创建电路](#创建电路)
- [读取电路信息](#读取电路信息)
- [测量记录采样](#测量记录采样)
- [Detector采样](#Detector采样)
- [逻辑错误统计](#逻辑错误统计)
- [编译并复用 Sampler](#编译并复用-sampler)
- [随机数与可复现性](#随机数与可复现性)
- [批处理和性能参数](#批处理和性能参数)
- [支持的 Stim 操作](#支持的-stim-操作)
- [完整 API 参考](#完整-api-参考)
- [异常、边界和线程行为](#异常边界和线程行为)
- [开发与测试](#开发与测试)

## 安装与构建

Python 扩展会直接编译仓库中的 C++ 源码。macOS/Linux 需要 C++20 编译器，
Windows 需要相应版本的 MSVC Build Tools；本次验证环境为 macOS。

推荐在本目录中使用虚拟环境和可编辑安装：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

普通本地安装：

```bash
python -m pip install .
```

仅在源码目录内构建并测试，不安装：

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

确认安装和 SIMD 后端：

```python
import symft

print(symft.__version__)
print(symft.active_simd_backend())
print(symft.active_batch_backend())
```

后端名称取决于编译目标和运行机器；业务逻辑不应依赖特定名称。

## 快速开始

```python
import symft

circuit = symft.Circuit("""
H 0
M 0
""")

samples = circuit.sample(shots=10, seed=123)
print(samples.shape)  # (10, 1)
print(samples.dtype)  # bool
```

多次对同一电路采样时，先编译 sampler：

```python
sampler = circuit.compile_sampler(batch=True)

first = sampler.sample(shots=1000, seed=1)
second = sampler.sample(shots=1000, seed=2)
```

只关心 detector 丢弃率和某个逻辑 observable 的错误率时，使用 counts API，
它不会把全部逐 shot 测量结果返回给 Python：

```python
result = circuit.sample_counts(shots=100_000, observable=0, threads=4)
print(result["discard_rate"])
print(result["logical_error_rate"])
```

## 创建电路

### 从文本创建

`Circuit` 接受 `str` 或 `bytes`。不传参数等价于空电路。

```python
circuit = symft.Circuit("X 0\nM 0\n")
empty = symft.Circuit()
```

### 从文件创建

`path` 支持 `str`、`bytes` 和 `os.PathLike`，因此可以直接传
`pathlib.Path`：

```python
from pathlib import Path
import symft

path = Path("circuits/d5.stim")
circuit = symft.Circuit(path=path)

# 等价的便捷函数
circuit = symft.read_stim_file(path)
```

`text` 与 `path` 互斥。解析失败、文件不存在或文件不可读都抛出
`symft.SymFTError`。`Circuit` 创建后不可通过再次调用 `__init__` 改写；需要
加载新电路时请创建新对象。

### 模块级便捷采样

`symft.sample` 接受电路文本或现有 `Circuit`，其余参数转发给
`Circuit.sample`：

```python
samples = symft.sample("X 0\nM 0\n", shots=8, seed=7)
```

文件路径不会被自动识别为路径，应使用 `read_stim_file` 或
`Circuit(path=...)`。

## 读取电路信息

```python
circuit = symft.Circuit("""
M 0
DETECTOR(1.5, 2.5) rec[-1]
OBSERVABLE_INCLUDE(2) rec[-1]
""")

print(circuit.num_qubits)                # 1
print(circuit.num_measurements)           # 1
print(circuit.num_detectors)              # 1
print(circuit.num_observables)            # 3，最大 observable 索引 + 1
print(circuit.num_observable_includes)     # 1，include 指令条数
print(circuit.detectors)
print(circuit.observables)
```

`detectors` 中每个字典包含：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `records` | `tuple[int, ...]` | detector 引用的测量记录编号，内部编号从 1 开始 |
| `coords` | `tuple[float, ...]` | 应用 `SHIFT_COORDS` 后的坐标 |
| `line` | `int` | 源电路中的行号，从 1 开始 |

`observables` 中每个字典包含 `index`、`records` 和 `line`。这里的
`records` 同样是内部 1-based 记录编号，不是 Stim 的负偏移写法。

## 测量记录采样

调用签名：

```python
circuit.sample(
    shots=1,
    seed=1,
    batch=False,
    bit_packed=False,
    batch_size=0,
    sample_chunk_shots=0,
)
```

普通输出为二维 `numpy.ndarray[numpy.bool_]`，形状固定为
`(shots, circuit.num_measurements)`。列顺序是电路产生测量记录的顺序。

```python
circuit = symft.Circuit("X 0\nM 0\nM 1\n")
samples = circuit.sample(shots=4, seed=1)

assert samples.shape == (4, 2)
assert samples[:, 0].all()
```

`shots=0` 仍保留正确列宽，例如 `(0, num_measurements)`。`shots` 必须在
`0..2**31-1` 范围内。

### 位压缩输出

`bit_packed=True` 返回 `numpy.uint8`，形状为
`(shots, ceil(num_measurements / 8))`。每个字节采用低位优先：测量记录 0
位于 byte 0 的 bit 0，记录 7 位于 bit 7，记录 8 位于 byte 1 的 bit 0。

```python
packed = circuit.sample(shots=1000, bit_packed=True)

# 恢复第 0 个测量记录
record_0 = (packed[:, 0] & 0x01) != 0
```

大规模逐 shot 输出优先使用位压缩，可将 Python 结果数组大小降低约 8 倍。
当前原生层在生成结果时仍会持有中间 packed words，因此极大 `shots × records`
任务应分块采样。

## Detector采样

```python
detectors = circuit.sample_detectors(
    shots=1000,
    seed=1,
    bit_packed=False,
)
```

普通输出形状为 `(shots, circuit.num_detectors)`，dtype 为 `numpy.bool_`。
`True` 表示对应 detector 在该 shot 触发。位压缩规则与测量记录一致，列宽为
`ceil(num_detectors / 8)`。

`sample_detectors` 当前使用 single-shot detector 执行路径，不接受
`batch`、`batch_size` 或 `threads` 参数。需要高吞吐 detector/逻辑错误聚合时，
使用 `sample_counts`。

## 逻辑错误统计

调用签名：

```python
circuit.sample_counts(
    shots=1,
    seed=1,
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
)
```

对每个 shot：任意 detector 触发则计入 `discarded`，否则计入 `accepted`；
选定 `observable` 的所有 `OBSERVABLE_INCLUDE` 记录按异或奇偶合并，结果为 1
的 accepted shot 计入 `logical_errors`。

返回字典：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `shots` | `int` | 请求并完成的 shot 数 |
| `discarded` | `int` | 任意 detector 触发的 shot 数 |
| `accepted` | `int` | 未触发 detector 的 shot 数 |
| `logical_errors` | `int` | accepted 中逻辑 observable 奇偶为 1 的数量 |
| `discard_rate` | `float` | `discarded / shots` |
| `logical_error_rate` | `float` | `logical_errors / accepted` |
| `active_threads` | `int` | 本次实际使用的 worker 数 |
| `timing` | `dict` | `parse_s`、`plan_s`、`presample_s`、`execute_s`、`accumulate_s`、`sample_s` |

分母为 0 时，相应 rate 是 `nan`。始终满足
`accepted + discarded == shots`。

```python
import math

result = circuit.sample_counts(shots=10_000, observable=0)
if not math.isnan(result["logical_error_rate"]):
    print(result["logical_error_rate"])
```

`observable` 是非负索引。若电路没有该索引的 `OBSERVABLE_INCLUDE`，逻辑错误
数会是 0；调用方应结合 `circuit.observables` 检查所需索引确实存在。

### Detector 后选择

`postselect_detectors=True` 允许 batch 执行在 detector 触发时提前淘汰 shot，
结果统计语义不变，适合 detector 较早出现且丢弃率较高的电路。

`batch_mask_threshold_denominator` 控制后选择中 dead-shot 压缩阈值，默认 2
表示 dead shots 达到约 `1/2` 时允许压缩。它只在
`batch=True, postselect_detectors=True` 时有效，通常保留默认值即可。

## 编译并复用 Sampler

### `CompiledMeasurementSampler`

```python
sampler = circuit.compile_sampler(
    batch=True,
    batch_size=0,
    sample_chunk_shots=0,
)

samples = sampler.sample(shots=1000, seed=1, bit_packed=False)
detectors = sampler.sample_detectors(shots=1000, seed=1)

print(sampler.num_qubits)
print(sampler.num_measurements)
print(sampler.num_detectors)
print(sampler.max_active_qubits)
```

它复用已经 lowering/planning 的 instruction program，适合对同一电路多次返回
逐 shot 数组。编译时的 `batch` 决定测量采样执行路径；其
`sample_detectors` 仍使用 single-shot detector 路径。

### `CompiledCountsSampler`

```python
sampler = circuit.compile_counts_sampler(
    batch=True,
    observable=0,
    postselect_detectors=True,
    threads=4,
)

print(sampler.info)
print(sampler.preprocessing_timing)

result_a = sampler.sample(shots=100_000, stream_id=10)
result_b = sampler.sample(shots=100_000, stream_id=11)
```

该 sampler 复用 instruction program、预采样计划和 worker 内存，适合重复执行
统计任务。`info` 给出规范化后的配置：

- `num_qubits`、`num_measurements`、`num_detectors`；
- `num_observable_includes`、`observable`；
- `max_active_qubits`；
- 最终 `batch_size`、`sample_chunk_shots`、`threads`；
- `detector_postselection` 和 `batch_mask_threshold_denominator`。

同一 `CompiledCountsSampler` 可以从多个 Python 线程调用，封装会串行执行这些
调用以保护其内部可变运行时。需要真正并行的独立任务时，应为每个调用线程
创建单独 sampler，或直接使用一个 sampler 的 `threads` 参数。

## 随机数与可复现性

逐 shot 数组 API 使用 `seed`：在相同版本、相同调用参数和相同执行路径下，
相同 seed 可复现结果。

```python
a = circuit.sample(shots=100, seed=123)
b = circuit.sample(shots=100, seed=123)
```

`Circuit.sample_counts(..., seed=N)` 使用 `N` 作为 prepared counts sampler 的
显式随机流标识。`CompiledCountsSampler.sample` 改用 `stream_id`：

- 显式传相同 `stream_id`，可复现相同统计结果；
- 省略 `stream_id`，sampler 从 0 开始自动递增内部 stream counter；
- 不应依赖 single-shot 与 batch 路径在同一标识下逐 bit 完全一致；
- 修改 batch/chunk 参数可能改变随机数分块，因此跨配置只比较统计分布。

## 批处理和性能参数

### `batch`

- 测量数组 API 默认 `False`，短任务和调试可保留默认值；
- counts API 默认 `True`，适合大规模聚合；
- `batch=False` 时，counts 的 `threads` 和 `batch_size` 不会增加并行度。

### `batch_size`

`0` 表示自动选择。当前自动值为：

```text
min(2048, max(1, 32768 / 2^max_active_qubits))
```

增大 batch 可能提高吞吐，也会按 active state 大小增加内存；高
`max_active_qubits` 电路应优先使用自动值。显式值必须为非负整数。

### `sample_chunk_shots`

`0` 表示自动选择，当前基础默认值为 2048；batch counts 会确保 chunk 不小于
batch。chunk 是预采样和线程任务划分单位。使用多个 `threads` 时，shot 数应有
足够多 chunk，否则 `active_threads` 可能小于请求值。

### `threads`

仅 batch counts 路径使用。`threads=1` 为默认值，`threads=0` 会被规范化为 1。
C++ 执行期间会释放 Python GIL。优先用 `sampler.info` 和结果中的
`active_threads` 确认最终配置与实际并行度。

### 选择接口

| 需求 | 建议接口 |
| --- | --- |
| 一次性获取测量 bit | `Circuit.sample` |
| 重复获取测量 bit | `compile_sampler().sample` |
| 获取每个 detector bit | `sample_detectors` |
| 只统计丢弃率/逻辑错误率 | `sample_counts` |
| 重复大规模统计 | `compile_counts_sampler().sample` |
| 降低返回数组内存 | `bit_packed=True` |

## 支持的 Stim 操作

接口解析 Stim 风格文本，但不是 Stim 全语法的透明替代。当前主要支持：

- 元数据和结构：`REPEAT`、`TICK`、`QUBIT_COORDS`、`SHIFT_COORDS`、
  `DETECTOR`、`OBSERVABLE_INCLUDE`；
- 单比特 Clifford：`I`、`H` 及 H/C 轴变体、`S`/`S_DAG`、
  `SQRT_X/Y/Z` 及 dagger、`X`、`Y`、`Z`；
- 双比特 Clifford：`CX`/`CNOT`、`CY`、`CZ`、`SWAP`、`ISWAP`、
  `SQRT_XX/YY/ZZ` 及 dagger，以及解析器中的轴控制变体；
- 非 Clifford/旋转：`T`、`T_DAG`、`R_X/Y/Z`、`R_XX/YY/ZZ`、
  `R_PAULI`、`U`/`U3`、`SPP`/`SPP_DAG`；
- 测量和重置：`M`/`MZ`、`MX`、`MY`、`MR*`、`R`/`RX/RY/RZ`、
  `MPP`、`MXX/MYY/MZZ`、`MPAD`；
- 噪声：`X/Y/Z_ERROR`、`I/II_ERROR`、`DEPOLARIZE1/2/3`、
  `PAULI_CHANNEL_1/2/3`、`CORRELATED_ERROR`/`E`、
  `ELSE_CORRELATED_ERROR`、`HERALDED_ERASE`、
  `HERALDED_PAULI_CHANNEL_1`；
- 测量记录控制的 Pauli feedback，以及部分 sweep-controlled Clifford 操作。

操作参数、target 数量和可用控制方向仍受解析器校验。遇到未支持操作会抛出
`SymFTError`，错误信息包含具体操作或约束。

## 完整 API 参考

### 模块级成员

```python
symft.__version__ -> str
symft.read_stim_file(path) -> Circuit
symft.sample(circuit, shots=1, **kwargs) -> numpy.ndarray
symft.active_simd_backend() -> str
symft.active_batch_backend() -> str
```

### `Circuit`

```python
symft.Circuit(text=None, path=None)

Circuit.compile_sampler(
    batch=False, batch_size=0, sample_chunk_shots=0
) -> CompiledMeasurementSampler

Circuit.compile_counts_sampler(
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
) -> CompiledCountsSampler

Circuit.sample(
    shots=1,
    seed=1,
    batch=False,
    bit_packed=False,
    batch_size=0,
    sample_chunk_shots=0,
) -> numpy.ndarray

Circuit.sample_counts(
    shots=1,
    seed=1,
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
) -> dict

Circuit.sample_detectors(
    shots=1, seed=1, bit_packed=False
) -> numpy.ndarray
```

只读属性：`num_qubits`、`num_measurements`、`num_detectors`、
`num_observables`、`num_observable_includes`、`detectors`、`observables`。

### `CompiledMeasurementSampler`

```python
sampler.sample(shots=1, seed=1, bit_packed=False) -> numpy.ndarray
sampler.sample_detectors(shots=1, seed=1, bit_packed=False) -> numpy.ndarray
```

只读属性：`num_qubits`、`num_measurements`、`num_detectors`、
`max_active_qubits`。

### `CompiledCountsSampler`

```python
sampler.sample(shots=1, stream_id=None) -> dict
```

只读属性：`info`、`preprocessing_timing`。

IDE 和静态类型检查器可读取包内 `.pyi` 与 `py.typed` 文件；运行时也可以用
`help(...)` 和 `inspect.signature(...)` 查看原生方法签名。

## 异常、边界和线程行为

- Stim 解析和大多数 C++ 运行时错误映射为 `symft.SymFTError`，它继承
  `RuntimeError`；
- Python 参数类型错误抛 `TypeError`；负数参数抛 `ValueError`；超出 C++
  `int` 范围抛 `OverflowError`；
- `seed` 和 `stream_id` 必须是非负、可转换为 `uint64` 的整数；
- `shots=0` 合法，rate 分母为 0 时返回 `nan`；
- 原生计算通常释放 GIL，但一个 `CompiledCountsSampler` 的并发调用会由封装
  串行化；
- `Circuit` 和 compiled sampler 都应视为不可变配置对象，不要手动再次调用
  `__init__`；
- 返回的 metadata 列表和字典是新建 Python 对象，修改它们不会改写电路。

建议在服务层捕获：

```python
try:
    circuit = symft.read_stim_file("circuit.stim")
except (TypeError, symft.SymFTError) as exc:
    print(f"cannot load circuit: {exc}")
```

## 开发与测试

运行完整测试：

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行示例：

```bash
PYTHONPATH=src python examples/basic.py
PYTHONPATH=src python examples/test_stim.py path/to/circuit.stim --shots 10000 --threads 4
```

构建 wheel 和源码包：

```bash
python -m pip wheel . --no-deps -w dist
python setup.py sdist
```

Python 目录依赖仓库中的 `../cpp/src`；生成的 sdist 会把该 C++ 源码树复制进
归档，以便从源码包独立构建。编译器出现“loop not vectorized”警告不代表构建
失败，最终使用的后端应通过 `active_simd_backend()` 查询。

