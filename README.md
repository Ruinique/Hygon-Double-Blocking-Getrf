<!--
 * @Descripttion: 
 * @Author: Ruinique
 * @version: 
 * @Date: 2025-03-30 17:21:56
 * @LastEditors: Ruinique
 * @LastEditTime: 2025-03-30 17:37:46
-->
# Hygon-Getrf
在海光 Z100 DCU 上特化实现的 LU 分解 (getrf)

## 功能概述

本项目提供了以下主要功能：
1. **LU 分解**：通过 GPU 加速实现高效的矩阵分解。
2. **命令行参数解析**：使用 `parseArgs` 函数解析用户输入的参数，支持灵活的配置。

## 使用方法

### 编译项目

确保已安装 CMake 和 CUDA 工具链，然后运行以下命令编译项目：

```bash
mkdir build && cd build
cmake -B . -S ..
make
```

### 运行程序

编译完成后，生成的可执行文件位于 `build/` 目录下。运行程序时可以通过命令行参数指定配置，例如：

```bash
// n 是矩阵单边规模
./getrf $n
```

### 更多命令行参数说明

```bash
./getrf $n $k $nb -p -c -v
```
分别是方阵的边长，累计块大小和分块大小。
如果有 `-p` 代表启用 pivoting
如果有 `-c` 代表启用 compare with cusolver
如果有 `-v` 代表校验后向误差