"""
    该文件用作将原先的数据集变换成 LLaMA-Factory 支持的数据格式。
    参考 LLaMA-Factory的 ./data/mllm_demo.py 写成
"""

import json
from utils.prompts import getUserPrompt
# import logging

if __name__ == '__main__':
    
    languageChoice = ["en", "zh"]
    for i, srcLang in enumerate(languageChoice):
        tgtLang = languageChoice[1-i]
        with open(f"./data/train_{srcLang}.json", 'r') as f:
            allData = json.load(f)
        for promptLang in languageChoice:
            LFFormatData = []
            for clipData in allData:
                videoID = clipData["video_id"]
                clipID = clipData["clip_id"]
                userPrompt = "<image>" + getUserPrompt(promptLang, srcLang, tgtLang, \
                    clipData[f"{srcLang.upper()}_sentence"], 0, dataset_type="image-text")
                LFDataItem = dict()
                LFDataItem["messages"] = [ {"content":userPrompt ,"role": "user"}, \
                    {"content": clipData[f"{tgtLang.upper()}_sentence"], "role": "assistant"}]
                LFDataItem["images"] = [f"frames/{videoID}/{videoID}_{clipID}_mid.jpg"]
                LFFormatData.append(LFDataItem)
            with open(f"./LLaMA-Factory/data/sl-{srcLang}-pl-{promptLang}-mid.json", 'w', encoding='utf-8') as f:
                json.dump(LFFormatData, f, ensure_ascii=False, indent=2)