from transformers import AutoProcessor
from LLM_vmt_dataset.vmtDataset import Dataset4SelectorTrain  # 假设这是测试数据集的类
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os
import logging
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from model.selector import selectorConfig, selector

logger = logging.getLogger('evaluateSelector')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)

def selector_collate_fn(batchRawData):
    batchInputs, batchVideoClipID = [], []
    for item in batchRawData:
        videoClipID = item['videoClipID']
        for i in range(5):
            batchInputs.append([
                {"role": "system", "content": ""},
                {"role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"/root/autodl-tmp/frames/50frames/{videoClipID}/{videoClipID}_{int(item['clusteredInfo'][i])}.jpg",
                        "max_pixels": 360 * 420,
                    },
                {"type": "text", "text": item['sentence']},],
                }
            ])
        
        # 添加纯文本
        batchInputs.append([
            {"role": "system", "content": ""},
            {"role": "user",
            "content": [{"type": "text", "text": item['sentence']}],
            }
        ])
        batchVideoClipID.append(videoClipID)
    return batchInputs, batchVideoClipID


def evaluate(args):
    """
    评估模型主函数
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    
    config = selectorConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pairNum=6  # 设置为6个图文对
    )
    
    model = selector.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained(
        './checkpoint/selectorInit',
        trust_remote_code=True
    )
    
    model.to(device)
    
    dataset = Dataset4SelectorTrain(args.test_data_path)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=selector_collate_fn)
    
    maxIndex, videoClipID = [], []
    selectorScoreRecords = []
    
    # 添加batch计数器
    batch_counter = 0
    
    for batch_data in tqdm(dataLoader):
        batchInputs, batchVideoClipID = batch_data
        imageInputs, videoInputs = process_vision_info(batchInputs)
        texts = [processor.apply_chat_template(itemInputs, tokenize=False, add_generation_prompt=False) for itemInputs in batchInputs]
    
        batchInputs = processor(
            text=texts,
            images=imageInputs,
            videos=videoInputs,
            padding=True,
            return_tensors="pt",
        )
        batchInputs = batchInputs.to(device)
        outputs = model(**batchInputs)
        scores = outputs.logits
        raw_scores = scores.detach().float().cpu().numpy()
        for clip_id, row in zip(batchVideoClipID, raw_scores.tolist()):
            visual_scores = row[:5]
            best_visual_score = max(visual_scores)
            best_visual_index = visual_scores.index(best_visual_score)
            text_score = row[5]
            selectorScoreRecords.append({
                "clipID": clip_id,
                "selector_scores": [float(x) for x in row],
                "best_visual_index": int(best_visual_index),
                "text_index": 5,
                "best_index": int(best_visual_index if best_visual_score > text_score else 5),
                "best_visual_score": float(best_visual_score),
                "text_score": float(text_score),
                "visual_gain": float(best_visual_score - text_score),
            })
        scores = raw_scores
        # 修改argmax操作，当有多个最大值时的选择逻辑
        max_values = np.max(scores, axis=1, keepdims=True)
        # 创建掩码，标识所有等于最大值的位置
        mask = (scores == max_values)
        # 对每一行，检查最大值索引是否包含5，如果包含则选5，否则随机选择
        random_indices = np.zeros(scores.shape[0], dtype=np.int64)
        for i in range(scores.shape[0]):
            # 获取该行所有最大值的索引
            max_indices = np.where(mask[i])[0]
            # 如果最大值索引包含5，直接选择5
            if 5 in max_indices:
                random_indices[i] = 5
            else:
                # 否则随机选择一个索引
                random_indices[i] = np.random.choice(max_indices)
        scores = random_indices
        scores = scores.tolist()
        maxIndex.extend(scores)
        videoClipID.extend(batchVideoClipID)
        
        logger.info(f"Batch max index: {scores}")
    
    # 将maxIndex和videoClipID保存为json文件
    assert len(maxIndex) == len(videoClipID)
    chooseOutput = [{"clipID": videoClipID[i], "picIDChoose": maxIndex[i]} for i in range(len(videoClipID))]
    with open(os.path.join(args.output_dir, "chooseOutput.json"), "w") as f:
        json.dump(chooseOutput, f, indent=4, ensure_ascii=False)

    chooseByClip = {item["clipID"]: item["picIDChoose"] for item in chooseOutput}
    for item in selectorScoreRecords:
        item["picIDChoose"] = int(chooseByClip[item["clipID"]])
        item["best_index"] = int(item["picIDChoose"])
    with open(os.path.join(args.output_dir, "selector_scores.json"), "w", encoding="utf-8") as f:
        json.dump(selectorScoreRecords, f, indent=4, ensure_ascii=False)

    logger.info(f"评估结果已保存到 {args.output_dir}")

    if args.scores_file_path is not None:
        from codes.fastEval import fastEval

        with open(args.scores_file_path, "r") as f:
            scoresOfClips = json.load(f)
        BLEUScore, COMETScore, BLEURTScore = fastEval(scoresOfClips, chooseOutput)
        logger.info(f"BLEUScore: {BLEUScore}")
        logger.info(f"COMETScore: {COMETScore}")
        logger.info(f"BLEURTScore: {BLEURTScore}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 selector 模型")
    
    # 数据集相关参数
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--scores_file_path", type=str, default=None, help="scores文件路径")
    # 模型相关参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, default=None, help="评估结果保存路径")
    
    # 评估相关参数
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'./eval/0-selector/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # log 设置
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "evaluate.log"))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # 开始评估
    evaluate(args)
    
    # 读取模型路径的父文件夹的父文件夹下的train.log内容
    try:
        train_log_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "train.log")
        if os.path.exists(train_log_path):
            logger.info("发现训练日志文件，记录内容如下：")
            with open(train_log_path, "r", encoding="utf-8") as f:
                train_log_content = f.read()
                logger.info(f"训练日志内容：\n{train_log_content}")
        else:
            logger.info(f"未找到训练日志文件：{train_log_path}")
    except Exception as e:
        logger.error(f"读取训练日志文件时出错：{str(e)}")
    
