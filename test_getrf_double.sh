#!/bin/bash

# 设置测试参数
start_n=8192
end_n=40960
step_n=1024
output_csv="getrf_results.csv"

# 获取脚本所在的目录，并切换到 float/build 目录
script_dir=$(dirname "$(realpath "$0")")
build_dir="$script_dir/double/build"

# 检查 build 目录是否存在
if [ ! -d "$build_dir" ]; then
  echo "错误: 目录 $build_dir 不存在。"
  echo "请先在 double 目录下运行编译命令:"
  echo "mkdir build && cd build"
  echo "cmake -B . -S .."
  echo "make"
  exit 1
fi

# 检查 getrf 可执行文件是否存在
executable="$build_dir/getrf"
if [ ! -x "$executable" ]; then
  echo "错误: 可执行文件 $executable 不存在或没有执行权限。"
  echo "请确保编译成功。"
  exit 1
fi

echo "切换到目录: $build_dir"
cd "$build_dir" || exit 1

# 创建/清空 CSV 文件并写入表头
echo "n,Double Blocking Time (ms),Double Blocking TFLOPS,cuSolver Time (ms)" > "$output_csv"
echo "将结果写入到: $(pwd)/$output_csv"

echo "开始测试 Hygon Getrf (double)..."

# 循环测试不同的矩阵规模 n
for (( n=start_n; n<=end_n; n+=step_n )); do
  echo "----------------------------------------"
  echo "运行测试: n = $n"
  echo "命令: ./getrf $n"
  # 执行命令并捕获输出
  output=$(./getrf "$n")
  echo "$output" # 打印原始输出到终端

  # 解析输出并提取数据
  # 使用 awk 进行更健壮的解析
  cusolver_time=$(echo "$output" | awk '/cusolver lu time:/ {print $4}')
  db_time=$(echo "$output" | awk '/double-blocking LU:/ && /ms/ {print $3}')
  db_tflops=$(echo "$output" | awk '/double-blocking LU:/ && /TFLOPS/ {print $3}')
  speed_up=$(echo "$output" | awk '/Speedup:/ {print $2}')

  # 检查是否成功提取到所有数据
  if [ -n "$cusolver_time" ] && [ -n "$db_time" ] && [ -n "$db_tflops" ] && [ -n "$speed_up" ]; then
    # 将数据追加到 CSV 文件
    echo "$n,$db_time,$db_tflops,$cusolver_time,$speed_up" >> "$output_csv"
    echo "数据已追加到 $output_csv"
  else
    echo "警告: 未能从 n=$n 的输出中完整提取数据，跳过写入 CSV。"
    echo "提取到的数据: cusolver_time='$cusolver_time', db_time='$db_time', db_tflops='$db_tflops'"
  fi

  echo "----------------------------------------"
  echo ""
done

echo "测试完成。结果保存在 $build_dir/$output_csv"

# 可选：切换回原来的目录
# cd "$script_dir"
