import json
import random
import argparse
import os

def process_data(train_all_file, train_clustered_file, out_train_file, out_clustered_file, begin, end):
    # 加载数据
    with open(train_all_file, 'r', encoding='utf-8') as f:
        train_all = json.load(f)
    with open(train_clustered_file, 'r', encoding='utf-8') as f:
        train_clustered = json.load(f)

    # 获取所有 cluster 中的 clip id
    clustered_ids = {list(item.keys())[0] for item in train_clustered}
    
    # 固定随机种子并打乱数据
    random.seed(42)
    random.shuffle(train_all)
    
    # 切片及过滤
    slice_size = 10000
    train_all = train_all[begin * slice_size: end * slice_size]
    train_all = [item for item in train_all if f"{item['video_id']}_{item['clip_id']}" in clustered_ids]

    # 保存过滤后的训练数据
    os.makedirs(os.path.dirname(out_train_file), exist_ok=True)
    with open(out_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_all, f, indent=4, ensure_ascii=False)
    
    # 构建 clip id 到 cluster 信息的映射，并重构 cluster 数据
    clip_to_cluster = {list(item.keys())[0]: list(item.values())[0] for item in train_clustered}
    train_clustered_data = [
        {f"{item['video_id']}_{item['clip_id']}": clip_to_cluster[f"{item['video_id']}_{item['clip_id']}"]}
        for item in train_all
    ]

    # 保存重构后的 cluster 数据
    os.makedirs(os.path.dirname(out_clustered_file), exist_ok=True)
    with open(out_clustered_file, 'w', encoding='utf-8') as f:
        json.dump(train_clustered_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理训练数据并生成 cluster 文件")
    parser.add_argument('--begin', type=int, default=0, help="起始索引乘数（每份数据10000条）")
    parser.add_argument('--end', type=int, default=20, help="结束索引乘数（每份数据10000条）")
    args = parser.parse_args()

    # 处理英文数据
    process_data(
        train_all_file='/root/autodl-tmp/train_en_clips_03_29.json',
        train_clustered_file='/root/autodl-tmp/frames/en_train_all_clustered.json',
        out_train_file='/root/autodl-tmp/work2/train_en_0_20.json',
        out_clustered_file='/root/autodl-tmp/work2/train_en_0_20_clustered.json',
        begin=args.begin, end=args.end
    )

    # 处理中文数据
    process_data(
        train_all_file='/root/autodl-tmp/train_zh_clips_03_29.json',
        train_clustered_file='/root/autodl-tmp/frames/legal_premix_zh_clustered.json',
        out_train_file='/root/autodl-tmp/work2/train_zh_0_20.json',
        out_clustered_file='/root/autodl-tmp/work2/train_zh_0_20_clustered.json',
        begin=args.begin, end=args.end
    )