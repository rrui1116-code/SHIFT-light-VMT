import logging
import pandas as pd
import random
import json
import os
from tqdm import tqdm

logger = logging.getLogger('evalModel')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


def save_to_json(df, output_path):
    # 将DataFrame转换为指定JSON格式
    result = []
    for _, row in df.iterrows():
        # 从后五个（前五个不是翻译对），随机选择一个相同索引的英文和中文描述
        index = random.randint(5, 9)
        en_sentence = row['enCap'][index]
        zh_sentence = row['chCap'][index]
        
        videoClipId = row['videoID']
        clip_start_second = videoClipId[12:].split('_')[0]
        clip_end_second = videoClipId[12:].split('_')[1]
        
        # 创建JSON结构
        item = {
            "videoClipId": videoClipId,
            "EN_sentence": en_sentence,
            "ZH_sentence": zh_sentence,
            "subtitle_start_second": 13, # 以下为了保持格式一致，无意义.
            "subtitle_end_second": 17,
            "clip_start_second": 10,
            "clip_end_second": 20,
        }
        result.append(item)

    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    logger.info(f"成功将{len(result)}条数据保存到JSON文件")
    return result


def check_frames_folders(base_path='./data/VATEX_ZH/50frames', target_count=51):
    """
    检查指定路径下的每个子文件夹是否包含指定数量的.jpg图片
    
    Args:
        base_path: 包含子文件夹的主目录路径
        target_count: 每个子文件夹应该包含的.jpg图片数量
        
    Returns:
        tuple: (是否所有文件夹都符合要求, 不符合要求的文件夹列表)
    """
    logger.info(f"开始检查{base_path}下的子文件夹")
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        logger.error(f"路径 {base_path} 不存在")
        return False, []
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    logger.info(f"找到 {len(subfolders)} 个子文件夹")
    
    all_valid = True
    invalid_folders = []
    
    # 检查每个子文件夹
    for folder in subfolders:
        folder_path = os.path.join(base_path, folder)
        jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        count = len(jpg_files)
        
        if count != target_count:
            all_valid = False
            invalid_folders.append((folder, count))
            logger.warning(f"文件夹 {folder} 包含 {count} 张图片，应为 {target_count} 张")
    
    # 输出结果
    if all_valid:
        logger.info(f"所有 {len(subfolders)} 个子文件夹都包含 {target_count} 张.jpg图片")
    else:
        logger.warning(f"发现 {len(invalid_folders)} 个不符合要求的文件夹")
        for folder, count in invalid_folders:
            logger.warning(f"  - {folder}: {count}张图片")
    
    return all_valid, invalid_folders


def cutVideos(results):
    success_count = 0
    os.makedirs("./data/VATEX_ZH/clips", exist_ok=True)
    
    for item in tqdm(results):
        videoClipId = item['videoClipId']
        clip_start_second = item['clip_start_second']
        
        sourceVideoPath = f"./data/VATEX_ZH/videos/{videoClipId}.mp4"
        
        try:
            os.system(f"ffmpeg -n -loglevel error -ss {clip_start_second} -i {sourceVideoPath}  -t 10 \
                    -c:v libx264 -c:a aac ./data/VATEX_ZH/clips/{videoClipId}.mp4")
            success_count += 1
        except Exception as e:
            logger.error(f"视频 {videoClipId} 剪辑失败: {e}")
    
    logger.info(f"成功剪辑 {success_count} 个视频")
    logger.info(f"失败剪辑 {len(results) - success_count} 个视频")




if __name__ == "__main__":
    
    fileHandler = logging.FileHandler(f'log/handleVatexData.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # 读取parquet文件
    df = pd.read_parquet('/home/byguan/LLMvmt/data/VATEX_ZH/vatex_val_zh/validation-00000-of-00001.parquet')    
    results = save_to_json(df, '/home/byguan/LLMvmt/data/VATEX_ZH/vatex_ZH_val.json')
    
    
    # cutVideos(results)
    
    # 检查frames文件夹
    # check_frames_folders()
    
    