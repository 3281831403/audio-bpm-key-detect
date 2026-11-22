# audio-bpm-key-detect

## 题面

素材库自动打 BPM 和调性标签后，速度不是一半就是两倍、调性还全错：起始包络不归一，bpm 偏高；自相关不除长度，长窗胜出；BPM 范围外不钳制，返回 300；半速/倍速不修正，随机取一个；色度用平均能量不分频带，调性随机；调性相关性用欧氏距离，小调被判大调；置信度恒为 1，误检无法过滤；样本过短不报错，自相关全 0；批量共用包络，轨之间串；hop=0 不校验，除零。bpmkey/analysis.py 是核心，tests/test_analysis.py 15个红。要修的是：起始包络归一化、自相关按重叠长度归一、BPM 范围钳制、半速/倍速按先验修正、按频带计算色度、用相关性判别大小调、置信度校准、短样本校验、批量包络隔离与整体回滚、hop 校验。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始（未实现）状态下必须恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error。输出尾部形如：

```text
15 failed in 0.05s
```

每个测试都直接调用 `bpmkey/analysis.py` 中的桩函数，当前失败原因是 `NotImplementedError`。实现完成后应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/test_analysis.py` 是验收测试，不可修改。不要通过改测试来让结果变绿。
- 仅使用 Python 标准库（`math`）与 `pytest`。离线运行，不使用随机数，不调用 `sleep`。
- 初始状态下 8 个待实现函数全部 `raise NotImplementedError`，因此 15 个测试应全部失败。
- 音频一律使用 `list[float]`，采样率显式传入。BPM 检测容差为 0.5 BPM，调性置信度为 0..1。
