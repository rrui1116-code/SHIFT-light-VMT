import argparse
import os
import logging
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel,AutoProcessor,AutoModel, BlipModel, Blip2Model,Blip2ForImageTextRetrieval,ChineseCLIPProcessor, ChineseCLIPModel
import random
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from LLM_vmt_dataset.vmtDataset import Dataset4Clip

# 日志设置
logger = logging.getLogger('extractFrames')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')


def getLanguage(args):
    with open(args.test_file_path, 'r') as f:
        test_data = json.load(f)
    if test_data[0]['language'] == 'en: English':
        src_language = 'en'
    elif test_data[0]['language'] == 'zh: Chinese':
        src_language = 'zh'
    else:
        raise ValueError("The src language is not specified and cannot be determined from the test file.")
    if args.src_language is None:
        return src_language
    else :
        assert args.src_language == src_language, "Check source language"
        return args.src_language

def buildModelAndProcessor(args):
    if args.model_name == "siglip-so400m-patch14-384":
        model = AutoModel.from_pretrained(args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path)
    elif args.model_name == "clip-vit-large-patch14":
        model = CLIPModel.from_pretrained(args.model_path)
        processor = CLIPProcessor.from_pretrained(args.model_path)
    elif args.model_name == "blip-image-captioning-base":
        model = BlipModel.from_pretrained(args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path)
    elif args.model_name == "blip2-itm-vit-g":
        model = Blip2ForImageTextRetrieval.from_pretrained(args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path)
    elif args.model_name == "chinese-clip-vit-base-patch16":
        model = ChineseCLIPModel.from_pretrained(args.model_path)
        processor = ChineseCLIPProcessor.from_pretrained(args.model_path)
    else:
        raise ValueError("Invalid model name")
    return model,processor

def extractBestFrame(args,dataset,logDirName):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,processor = buildModelAndProcessor(args)
    model.to(device)
    results = []
    infos = []
    for index,item in enumerate(tqdm(dataloader, desc="inferencing", unit="条")):
        text = item["text"][0]
        images = item["images"][0]
        padding = {"siglip-so400m-patch14-384": "max_length", "clip-vit-large-patch14": "max_length", "blip-image-captioning-base": True, "blip2-itm-vit-g": True,"chinese-clip-vit-base-patch16":True}
        max_length = {"siglip-so400m-patch14-384": 64, "clip-vit-large-patch14": 77, "blip-image-captioning-base": None, "blip2-itm-vit-g": None,"chinese-clip-vit-base-patch16":None}
        if max_length[args.model_name] is not None:
            inputs = processor(text=text, images=images, padding=padding[args.model_name],return_tensors="pt", truncation=True,max_length=max_length[args.model_name]).to(device)
        else:
            inputs = processor(text=text, images=images, padding=padding[args.model_name],return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_text # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # 获取最大值的索引
        max_value_index = torch.argmax(probs).item()

        # 将每个值和索引存入字典
        index_value_dict = {i: probs[0, i].item() for i in range(probs.size(1))}

        # 保存信息
        result = {
            "clipID": item["videoClipID"][0],
            "picIDChoose": max_value_index
        }
        info = {**index_value_dict, **result}
        results.append(result)
        infos.append(info)

    with open(f"{logDirName}/results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(f"{logDirName}/infos.json", "w") as f:
        json.dump(infos, f, indent=4, ensure_ascii=False)

def collate_fn(batchRawData):
    batchData = {"text":[], "images":[], "videoClipID":[]}
    for dataItem in batchRawData:
        batchData["text"].append(dataItem['text'])
        batchData["images"].append(dataItem['images'])
        batchData["videoClipID"].append(dataItem['videoClipID'])
    return batchData    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--clustered_file_path", type=str, default="./data/frames/test_general_en_clustered.json")
    parser.add_argument('-t', "--test_file_path", type=str, default="/home/share/vmtData/test/Test_general_en.json") # 测试集路径
    parser.add_argument('-f', "--frams_path", type=str, default="./data/frames/50frames")
    parser.add_argument('-m', "--model_path", type=str, default='/home/chan/openai/clip-vit-large-patch14')
    parser.add_argument('-s', "--src_language", type=str, default=None,choices=['zh', 'en'],help="source language(zh or en)") # 源语言,非必须使用该参数，后续可以通过json文件中的字段获取
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    # log 设置
    while True:
        logDirName = f'./eval/eval-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
        if not os.path.exists(logDirName):
            os.makedirs(logDirName)
            break
        else:
            time.sleep(random.randint(60, 240))

    fileHandler = logging.FileHandler(f'{logDirName}/eval.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    # 获取模型名称
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]

    # 确定源语言是英文/中文
    args.src_language = getLanguage(args)

    # 将参数记录到日志中
    args2Log = "Script arguments: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)

    # 加载数据集并创建DataLoader，目前只支持batch_size=1
    dataset = Dataset4Clip(args)
    dataloader = DataLoader(dataset, collate_fn=collate_fn ,batch_size=1, shuffle=False, num_workers=32,prefetch_factor=4)

    # # 加载模型并提取最佳帧，保存结果
    extractBestFrame(args,dataloader,logDirName)
