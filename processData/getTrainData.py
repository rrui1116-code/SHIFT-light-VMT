import json
import numpy as np
import os
import logging
import sys
import random
from utils.computeTransMetric import getSrcPredsRefs, computeCOMET

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler('./log/train_data_processing.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def custom_argmax(arr, axis=None):
    """
    自定义argmax函数，当第6个元素（索引5）是最大值时优先返回5，
    否则在多个最大值中随机选择一个索引。
    """
    # 获取原始数组中的最大值
    max_value = np.max(arr, axis=axis)
    
    # 如果是一维数组
    if axis is None or arr.ndim == 1:
        # 检查索引5的元素是否等于最大值
        if arr.size > 5 and arr[5] == max_value:
            return 5
        else:
            # 找出所有等于最大值的索引
            max_indices = np.where(arr == max_value)[0]
            # 随机选择一个索引
            return np.random.choice(max_indices)
    
    # 如果是多维数组（按行处理）
    if axis == 1:
        result = np.zeros(arr.shape[0], dtype=int)
        for i in range(arr.shape[0]):
            row = arr[i]
            # 检查索引5的元素是否等于该行的最大值
            if row.size > 5 and row[5] == max_value[i]:
                result[i] = 5
            else:
                # 找出所有等于最大值的索引
                max_indices = np.where(row == max_value[i])[0]
                # 随机选择一个索引
                result[i] = np.random.choice(max_indices)
        return result
    
    # 其他情况使用numpy原生argmax（可根据需要扩展）
    return np.argmax(arr, axis=axis)


def saveOrLoadCOMETScores(Dirs, translationDirect):
    save_dir = './data/work2/'
    baseName = os.path.basename(Dirs[0])
    # 提取en_zh_0_20部分
    base = '_'.join(baseName.split('_')[:-1])  # 去掉最后一个数字
    save_path = os.path.join(save_dir, f"train_{base}_comet_scores.npy")
    if os.path.exists(save_path):
        return np.load(save_path)
    else:
        srcPredsRefs = [ getSrcPredsRefs(DirName) for DirName in Dirs ]
        allCometScores = np.array([ computeCOMET(src, preds, refs)["scores"] for src, preds, refs in srcPredsRefs ])
        allCometScores = np.transpose(allCometScores)
        np.save(save_path, allCometScores)
        # print(f"COMET分数已保存到: {save_path}")
        return allCometScores

def getTrainData(allCometScores, language):
    
    with open(f/root/autodl-tmp/train_data/train_{language}_0_20_clustered.json", "r") as f:
        clustered = json.load(f)
    
    with open(f/root/autodl-tmp/train_data/train_{language}_0_20.json", "r") as f:
        clipData = json.load(f)
    
    clipDataFiltered, clusteredFiltered, cometScoresFiltered = [], [], []
    
    for i, (cometScores, clipInfo, clusteredInfo) in enumerate(zip(allCometScores, clipData, clustered)):
        assert f"{clipInfo['video_id']}_{clipInfo['clip_id']}" == list(clusteredInfo.keys())[0]
        
        # 如果最好的文本仍低于0.6，则认为整体翻译质量不佳，跳过
        maxScore = np.max(cometScores)
        if maxScore < 0.6:
            continue

        # 如果最好和最差的文本差距小于0.02，则认为差距很小，跳过
        minScore = np.min(cometScores)
        if maxScore - minScore < 0.02:
            continue
        
        # 如果高值下标有多个，且包含5，则认为5是最好的，给下标值为5的数多加0.01
        if np.sum(cometScores == maxScore) > 1 and 5 in np.where(cometScores == maxScore)[0]:
            cometScores[5] = min(cometScores[5] + 0.01, 1)
        
        # 创建一个新的clipInfo对象，只保留必要的键值对
        filtered_clipInfo = {
            "videoClipID": f"{clipInfo['video_id']}_{clipInfo['clip_id']}",
            "sentence": clipInfo[f'{clipInfo["language"].split(":")[0].strip().upper()}_sentence'],
            "cometScores": cometScores.tolist(),
            "clusteredInfo": list(clusteredInfo.values())[0]
        }
        
        cometScoresFiltered.append(cometScores)
        clipDataFiltered.append(filtered_clipInfo)
        clusteredFiltered.append(clusteredInfo)
    
    cometScoresFiltered = np.array(cometScoresFiltered)
    
    with open(f/root/autodl-tmp/train_data/train_{language}_0_20_filtered.json", "w") as f:
        json.dump(clipDataFiltered, f, indent=4, ensure_ascii=False)
    
    with open(f/root/autodl-tmp/train_data/train_{language}_0_20_clustered_filtered.json", "w") as f:
        json.dump(clusteredFiltered, f, indent=4, ensure_ascii=False)
    
    np.save(f/root/autodl-tmp/train_data/train_{language}_0_20_comet_scores_filtered.npy", cometScoresFiltered)

    return clipDataFiltered, clusteredFiltered, cometScoresFiltered

def createEvenDistribution(clipDataFiltered, cometScoresFiltered, language):
    """
    创建一个类别分布均衡的数据集，使第六类样本(argmax_indices==5)数量与其他五类的平均数量相同
    """
    # 使用custom_argmax确定每个样本的类别
    row_argmax_indices = custom_argmax(cometScoresFiltered, axis=1)
    
    # 统计每个类别的数量
    class_counts = [np.sum(row_argmax_indices == i) for i in range(6)]
    logging.info(f"{language} 各类样本原始数量: {class_counts}")
    
    # 计算前五类的平均数量
    avg_count_first_five = int(np.mean([class_counts[i] for i in range(5)]))
    logging.info(f"{language} 前五类样本平均数量: {avg_count_first_five}")
    
    # 为每个类别选择样本
    balanced_indices = []
    for class_idx in range(6):
        class_indices = np.where(row_argmax_indices == class_idx)[0]
        if class_idx == 5:  # 对第六类样本(索引5)
            # 保留与前五类平均数量相同的样本
            keep_count = avg_count_first_five
            # 确保keep_count不超过class_indices的长度
            keep_count = min(keep_count, len(class_indices))
            selected_indices = np.random.choice(class_indices, keep_count, replace=False)
        else:
            # 其他类别全部保留
            selected_indices = class_indices
        balanced_indices.extend(selected_indices)
    
    # 打乱顺序
    np.random.shuffle(balanced_indices)
    
    # 选择平衡后的数据
    balanced_clipData = [clipDataFiltered[i] for i in balanced_indices]
    balanced_cometScores = cometScoresFiltered[balanced_indices]
    
    # 统计平衡后各类数量
    balanced_argmax_indices = custom_argmax(balanced_cometScores, axis=1)
    balanced_counts = [np.sum(balanced_argmax_indices == i) for i in range(6)]
    logging.info(f"{language} 平衡后各类样本数量: {balanced_counts}")
    
    # 保存平衡后的数据
    with open(f/root/autodl-tmp/train_data/train_{language}_0_20_even.json", "w") as f:
        json.dump(balanced_clipData, f, indent=4, ensure_ascii=False)
    
    np.save(f/root/autodl-tmp/train_data/train_{language}_0_20_comet_scores_even.npy", balanced_cometScores)
    
    logging.info(f"已将平衡后的数据保存至: ./data/work2/train_{language}_0_20_even.json")
    
    return balanced_clipData, balanced_cometScores

if __name__ == "__main__":

    # 6个dir，分别是图片ID 0-4，和纯文本
    enDirs = [f'./eval/sets/differentPicID4Train/en_zh_0_20_{i}' for i in range(6)]
    zhDirs = [f'./eval/sets/differentPicID4Train/zh_en_0_20_{i}' for i in range(6)]

    cometScoresEn = saveOrLoadCOMETScores(enDirs, "en_zh_0_20")
    cometScoresZh = saveOrLoadCOMETScores(zhDirs, "zh_en_0_20")
    
    enClipDataFiltered, enClusteredFiltered, enCometScoresFiltered = getTrainData(cometScoresEn, "en")
    zhClipDataFiltered, zhClusteredFiltered, zhCometScoresFiltered = getTrainData(cometScoresZh, "zh")
    

    logging.info(f"enClipDataFiltered: {len(enClipDataFiltered):,}")
    logging.info(f"enClusteredFiltered: {len(enClusteredFiltered):,}")
    logging.info(f"enCometScoresFiltered: {enCometScoresFiltered.shape}")
    logging.info(f"zhClipDataFiltered: {len(zhClipDataFiltered):,}")
    logging.info(f"zhClusteredFiltered: {len(zhClusteredFiltered):,}")
    logging.info(f"zhCometScoresFiltered: {zhCometScoresFiltered.shape}")
    
    # 创建平衡数据集
    enClipDataBalanced, enCometScoresBalanced = createEvenDistribution(enClipDataFiltered, enCometScoresFiltered, "en")
    zhClipDataBalanced, zhCometScoresBalanced = createEvenDistribution(zhClipDataFiltered, zhCometScoresFiltered, "zh")
    
    # 创建合并的平衡数据集
    balancedTrainData = enClipDataBalanced + zhClipDataBalanced
    # 打乱数据
    random.shuffle(balancedTrainData)
    logging.info(f"打乱后的平衡数据集长度: {len(balancedTrainData):,}")
    
    # 保存合并的平衡数据集
    with open(/root/autodl-tmp/train_data/train_data_balanced.json", "w", encoding="utf-8") as f:
        json.dump(balancedTrainData, f, indent=4, ensure_ascii=False)
    logging.info(f"已将平衡后的合并数据保存至: ./data/work2/train_data_balanced.json")
    
    trainDataFiltered = enClipDataFiltered + zhClipDataFiltered
    # 打乱trainDataFiltered
    random.shuffle(trainDataFiltered)
    logging.info(f"打乱后的trainDataFiltered长度: {len(trainDataFiltered):,}")
    
    # 保存为JSON文件
    with open(/root/autodl-tmp/train_data/train_data_shuffled.json", "w", encoding="utf-8") as f:
        json.dump(trainDataFiltered, f, indent=4, ensure_ascii=False)
    logging.info(f"已将打乱后的数据保存至: ./data/work2/train_data_shuffled.json")
    
    enRow_argmax_indices = custom_argmax(enCometScoresFiltered, axis=1)
    zhRow_argmax_indices = custom_argmax(zhCometScoresFiltered, axis=1)

    # # 打印统计结果
    logging.info(f"统计结果:")
    for i in range(6):
        count = np.sum(enRow_argmax_indices == i)
        logging.info(f"en_0_20 选择{i}的次数: {count}")    
    
    logging.info(f"zh_0_20 选择统计：")
    for i in range(6):
        count = np.sum(zhRow_argmax_indices == i)
        logging.info(f"zh_0_20 选择{i}的次数: {count}")    
    
    # 从中随机挑选十个查看
    enCometScoresFiltered = enCometScoresFiltered[np.random.choice(len(enCometScoresFiltered), 10, replace=False)]
    zhCometScoresFiltered = zhCometScoresFiltered[np.random.choice(len(zhCometScoresFiltered), 10, replace=False)]
    
    logging.info(f"enCometScoresFiltered 随机挑选10个: {enCometScoresFiltered}")
    logging.info(f"zhCometScoresFiltered 随机挑选10个: {zhCometScoresFiltered}")
    