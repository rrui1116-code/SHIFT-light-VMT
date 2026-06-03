import json
import re
import os

from utils.computeTransMetric import computeTranslationMetrics

def extract_translated(text):
    # 使用正则表达式匹配 <translated> 与 </translated> 之间的内容
    match = re.search(r'<translation>(.*?)</translation>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'<translated>(.*?)</translated>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'<sentence>(.*?)</sentence>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

dirList = ['./eval/eval-2025-03-28-14-34', './eval/eval-2025-03-28-14-35', './eval/eval-2025-03-28-14-37', './eval/eval-2025-03-28-14-44']
dirList += ['./eval/eval-2025-03-28-14-51', './eval/eval-2025-03-28-14-54', './eval/eval-2025-03-28-14-56', './eval/eval-2025-03-28-14-58']

for dirPath in dirList:
    with open(f'{dirPath}/results.json', 'r') as f:
        data = json.load(f)
    for item in data:
        s = extract_translated(item['preds'])
        if s is not None:
            item['preds'] = s
    with open(f'{dirPath}/results.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    computeTranslationMetrics(dirPath)







