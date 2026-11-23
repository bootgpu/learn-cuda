# learn-cuda

[English](README.md) | 中文

[BootGPU](https://www.bootgpu.com/zh/) 课程 **CUDA 并行编程** 的题目仓库：一章一章写 CUDA C++ kernel，从 GPU 编程基础写到经典并行算法。

学这门课请打开[课程页](https://www.bootgpu.com/zh/courses/learn-cuda/)。每章有学习资料和若干关，关卡分测验和编程题；编程题在浏览器里写 kernel，提交后在 NVIDIA T4 上判分。测试规模很小（最多约一万个元素），写对就能过；怎么写快，是后续课程 Peak LLM Kernels 的内容。官方题解不在本仓库，通过一关后可在网站上查看。各章写完一章发布一章。

## 课程作者

### 目录结构

```text
course.yml                         课程目录：篇及其下的章，按顺序
chapters/<chapter>/
  chapter.yml                      本章标题与关卡，每关带状态
  reading.md, reading.zh.md        本章学习资料（英文 / 中文）
problems/<slug>/                   编程关
  problem.md, problem.zh.md        题面（英文 / 中文）
  problem.py                       参考实现、函数签名与测试用例
  starter/cuda/solution.cu         初始代码
  harness.cu                       可选：固定的 solve()，include solution.cu；判题时编译它
problems/<slug>/                   测验关
  quiz.yml, quiz.zh.yml            题目（英文 / 中文）
bootgpu/problem.py                 所有 problem.py 的基类
bootgpu.toml                       运行环境、GPU 与时限
scripts/                           validate.py、verify.py、build.py
tests/                             脚本的测试
archive/                           早期目录与退役题目，不上线
```

### 检查

```sh
uv sync
uv run scripts/validate.py   # 课程结构与全部题目，CPU 上运行
uv run pytest -q
```

`scripts/verify.py` 在 GPU 上给官方题解判分，需要私有题解仓库和判题密钥。

## 许可证

[CC BY-NC-ND 4.0](LICENSE)：可注明出处后分享，不可商用，不可发布改编版本。
