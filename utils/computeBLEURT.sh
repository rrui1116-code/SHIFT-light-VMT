#!/bin/zsh

export CUDA_VISIBLE_DEVICES=2


# 遍历./eval下的所有子目录，按名称升序排列，处理直到指定文件夹
find ./eval -maxdepth 1 -type d -mindepth 1 -print0 | sort -z | while IFS= read -r -d '' dir_path; do
    dir_name=$(basename "$dir_path")
    # 执行Python脚本

    echo "Processing $dir_name"
    python3 ./utils/computeTransMetric.py --dir_path "$dir_path"
    # 检查是否达到停止条件
    if [[ "$dir_name" == "eval-2025-03-10-15-24" ]]; then
        exit 0
    fi
done
