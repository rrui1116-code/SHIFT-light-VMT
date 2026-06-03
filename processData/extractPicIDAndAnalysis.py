# 分析一阶段推理结果的图片ID，并奖结果保存到json文件中

import json
import re
import os
import random
import logging 

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger('extractPreds')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def readLogInfo(logFilePath):
    # 定义需要提取的键
    keys = ['model_name', 'source_language', 'target_language', 'prompt_language']
    
    with open(logFilePath, 'r') as file:
        logContent = file.read()

    # 使用正则表达式提取信息
    logInfo = {}
    for key in keys:
        # 定义正则表达式
        pattern = rf'{key}: ([^\s]+)'  # 匹配键和值，中间是任意非空格字符
        match = re.search(pattern, logContent)
        if match:
            logInfo[key] = match.group(1)
    return logInfo

def handleLogDir(dirPath):
    with open(os.path.join(dirPath, 'results.json')) as f:
        picIDInThisFile = json.load(f)
    
    failPreds = []
    clipID2PicID = dict()
    
    failPreds = 0
    for item in picIDInThisFile:
        if item['preds'].isdigit() and 1 <= int(item['preds']) <= 10:
            predPicId = int(item['preds']) - 1
        else:
            failPreds += 1
            predPicId = random.randint(0, 9)
        clipID2PicID[item['clipID']] = predPicId
    logger.info(f'Fail preds: {failPreds}/7000')    
    
    logFilePath = os.path.join(dirPath, 'eval.log')
    thisFile = readLogInfo(logFilePath)
    
    thisFile['clipID2PicID'] = clipID2PicID
    return thisFile

def plotFigureOfDifferentSetting(picIDs):    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Four sets of choose image', fontsize=16)
    
    x = np.arange(1, 11)
    
    for subPlot in range(4):
        for i in range(3):
            settingData = [0]*10 
            thisSetting = picIDs[subPlot*3+i]['clipID2PicID']
            for _, value in thisSetting.items():
                settingData[value] += 1
            # 在第subplot个子图上绘制数据
            axs[subPlot//2, subPlot%2].plot(x, settingData, label=f'{i+1} th setting')
            print(f'{i+1} th setting: {settingData}')
        axs[subPlot//2, subPlot%2].set_xticks(x)
        infoOfSubPlot = picIDs[subPlot*3]
        axs[subPlot//2, subPlot%2].set_title(f'{infoOfSubPlot["source_language"]}_{infoOfSubPlot["target_language"]}_{infoOfSubPlot["prompt_language"]}')
        axs[subPlot//2, subPlot%2].legend()
    
    plt.tight_layout() # 调整子图之间的间距
    os.makedirs('./plot', exist_ok=True)
    plt.savefig('./plot/inferencePicID.pdf', format='pdf')

def plotFigureOfComparePicIDChange(picIDs, changePicIDs, modelName):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Change Picture ID Comparation of Model {modelName}', fontsize=16)
    
    x = np.arange(1, 11)
    
    # 用于存储差值统计的列表
    diffCountsFinal = []
    
    for subPlot in range(4):
        diff_counts = [0] * 10
        originalPicIDCount, changePicIDCount, changePicIDCountShift = [0]*10, [0]*10, [0]*10 
        originalClipID2PicID, picChangeClipID2PicID = picIDs[subPlot], changePicIDs[subPlot]
        valuesOfOriginal, valuesOfChange = list(originalClipID2PicID['clipID2PicID'].values()), list(picChangeClipID2PicID['clipID2PicID'].values())
        assert len(valuesOfOriginal) == len(valuesOfChange), "Length of valuesOfOriginal and valuesOfChange should be the same"
        for orginalValue, changeValue in zip(valuesOfOriginal, valuesOfChange):
            originalPicIDCount[orginalValue] += 1
            changePicIDCount[changeValue] += 1
            changePicIDCountShift[(changeValue+5)%10] += 1
            
            diff = abs((changeValue + 5) % 10 - orginalValue)
            diff_counts[diff] += 1
        
        diffCountsFinal.append(diff_counts)
        # 在第subplot个子图上绘制数据
        print(f'{subPlot+1} th setting: {originalPicIDCount}')
        print(f'{subPlot+1} th setting: {changePicIDCount}')
        axs[subPlot//2, subPlot%2].plot(x, originalPicIDCount, label=f'original')
        axs[subPlot//2, subPlot%2].plot(x, changePicIDCount, label=f'change')
        axs[subPlot//2, subPlot%2].plot(x, changePicIDCountShift, label=f'change shift')
        axs[subPlot//2, subPlot%2].set_xticks(x)
        axs[subPlot//2, subPlot%2].set_title(f'{picChangeClipID2PicID["source_language"]}_{picChangeClipID2PicID["target_language"]}_{picChangeClipID2PicID["prompt_language"]}')
        axs[subPlot//2, subPlot%2].legend()
    
    plt.tight_layout() # 调整子图之间的间距
    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f'./plot/{modelName}_self_infer_picID_distri.pdf', format='pdf')
    
    # 创建新的图形来绘制差值统计
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8))
    fig2.suptitle(f'Absloute Difference Statistics of Model {modelName} Self Inference PicID', fontsize=16)
    x = np.arange(10)
    
    for subPlot in range(4):
        
        picChangeClipID2PicID = changePicIDs[subPlot]
        
        diff_counts_subplot = diffCountsFinal[subPlot]
        axs2[subPlot//2, subPlot%2].plot(x, diff_counts_subplot, label=f'difference')
        axs2[subPlot//2, subPlot%2].set_xticks(x)
        axs2[subPlot//2, subPlot%2].set_title(f'{picChangeClipID2PicID["source_language"]}_{picChangeClipID2PicID["target_language"]}_{picChangeClipID2PicID["prompt_language"]}')
        # axs2[subPlot//2, subPlot%2].legend()
    
    plt.tight_layout()
    plt.savefig(f'./plot/{modelName}_self_infer_picID_abs_difference.pdf', format='pdf')
    
# 统计每种metric的max index 分布比例
def plotFigureOfMaxIndexDistribution(maxIndexFiles):
    # 两行（COMET, BLEU），两列（en->zh, zh->en）
    def getMaxIndex(maxIndexFilePath):
        with open(maxIndexFilePath, 'r') as f:
            clipID2PicID = json.load(f)
        max_COMET_Indies = [item["max_COMET_picID"] for item in clipID2PicID]
        max_BLEU_Indies = [item["max_BLEU_picID"] for item in clipID2PicID]
        return max_COMET_Indies, max_BLEU_Indies
    
    en_zh_max_COMET_Indies, en_zh_max_BLEU_Indies = getMaxIndex(maxIndexFiles[0])
    zh_en_max_COMET_Indies, zh_en_max_BLEU_Indies = getMaxIndex(maxIndexFiles[1])
    
    modelName = maxIndexFiles[0].split('/')[-1].split('_')[0]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Max Score Index Distribution of Model {modelName}', fontsize=16)
    x = np.arange(10)
    for i, metricName in enumerate(['COMET', 'BLEU']):
        for j, direction in enumerate(['en->zh', 'zh->en']):
            max_COMET_Indies, max_BLEU_Indies = (en_zh_max_COMET_Indies, en_zh_max_BLEU_Indies) if j == 0 else (zh_en_max_COMET_Indies, zh_en_max_BLEU_Indies)
            maxIndexCount = [0]*10
            for value in max_COMET_Indies if i == 0 else max_BLEU_Indies:
                maxIndexCount[value] += 1
            axs[i, j].plot(x, maxIndexCount)
            axs[i, j].set_xticks(x)
            axs[i, j].set_title(f'{direction} {metricName}')
            axs[i, j].set_xlabel('Max Score Index')
            axs[i, j].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'./plot/{modelName}_max_score_dis.pdf', format='pdf')
    
    # 统计每种metric的max index 绝对差值
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Max Score Of BLEU And COMET Index Abs Difference of Model {modelName}', fontsize=16)
    for i, direction in enumerate(['en->zh', 'zh->en']):
        max_COMET_Indies, max_BLEU_Indies = (en_zh_max_COMET_Indies, en_zh_max_BLEU_Indies) if i == 0 else (zh_en_max_COMET_Indies, zh_en_max_BLEU_Indies)
        assert len(max_COMET_Indies) == len(max_BLEU_Indies), "Length of max_COMET_Indies and max_BLEU_Indies should be the same"
        absDiff = [0]*10
        for COMETValue, BLEUValue in zip(max_COMET_Indies, max_BLEU_Indies):
            absDiff[abs(COMETValue - BLEUValue)] += 1
        axs[i].plot(x, absDiff)
        axs[i].set_xticks(x)
        axs[i].set_title(f'{direction} absolute difference')
        axs[i].set_xlabel('Absolute Difference')
        axs[i].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'./plot/{modelName}_max_score_abs_difference.pdf', format='pdf')

def plotWhetherCometBLEUSame(maxScoreIndexFiles):
    def getScores(maxIndexFilePath):
        with open(maxIndexFilePath, 'r') as f:
            clipID2PicID = json.load(f)
        cometScores, bleuScores = [item["COMET_scores"] for item in clipID2PicID], [item["BLEU_scores"] for item in clipID2PicID]
        return np.array(cometScores), np.array(bleuScores)

    modelName = maxScoreIndexFiles[0].split('/')[-1].split('_')[0]
    
    en_zh_COMET_scores, en_zh_BLEU_scores = getScores(maxScoreIndexFiles[0])
    zh_en_COMET_scores, zh_en_BLEU_scores = getScores(maxScoreIndexFiles[1])
    
    sameCount = [0, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'{modelName} Max Index Of BLEU And COMET Scores Same or Not', fontsize=16)
    
    for i, direction in enumerate(['en->zh', 'zh->en']):
        COMET_scores, BLEU_scores = (en_zh_COMET_scores, en_zh_BLEU_scores) if i == 0 else (zh_en_COMET_scores, zh_en_BLEU_scores)
        assert COMET_scores.shape == BLEU_scores.shape, "Shape of COMET_scores and BLEU_scores should be the same"
        for COMETValues, BLEUValues in zip(COMET_scores, BLEU_scores):
            max_COMET_value, max_BLEU_value = np.max(COMETValues), np.max(BLEUValues)
            close_COMET_indices = set(np.where(np.abs(COMETValues - max_COMET_value) < 0.01)[0]) # 0.01是一个COMET的阈值
            close_BLEU_indices = set(np.where(np.abs(BLEUValues - max_BLEU_value) < 0.1)[0]) # 0.1是一个BLEU的阈值
            
            if close_COMET_indices & close_BLEU_indices:
                sameCount[i] += 1
        logger.info(f'{modelName} {direction} same count: {sameCount[i]}/{len(COMET_scores)}, ratio is {round(sameCount[i]/len(COMET_scores)*100,2)}%')
        bars = axes[i].bar(['Same', 'Not Same'], [sameCount[i], len(COMET_scores) - sameCount[i]])
        axes[i].set_title(f'{direction}')
        axes[i].set_ylabel('Count')
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval, f"{yval}, {round(yval/len(COMET_scores)*100,2)}%", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(f'./plot/{modelName}_max_score_index_same_or_not.pdf', format='pdf')

if __name__ == '__main__':
    
    fileHandler = logging.FileHandler(f'./log/extractPicID.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # 按照 sl tl pl 分别是 zh en zh, zh en en, en zh zh, en zh en 的顺序做。
    qwenvl2OriginalDirs = ["./eval/eval-2025-02-12-22-25", "./eval/eval-2025-02-13-19-07", "./eval/eval-2025-02-13-07-07", "./eval/eval-2025-02-13-10-24"]
    qwenvl2changePicIDDirs = ["./eval/eval-2025-02-18-11-11", "./eval/eval-2025-02-18-11-12", "./eval/eval-2025-02-18-11-13", "./eval/eval-2025-02-18-20-08"]
    
    os.makedirs('./data/picSelect', exist_ok=True)
    
    picIDs = [handleLogDir(dirName) for dirName in qwenvl2OriginalDirs]
    with open('./data/picSelect/self_Inference_Qwen2-VL-7B-Instruct.json', 'w') as f:
        json.dump(picIDs, f, ensure_ascii=False, indent=4) 
    # plotFigureOfDifferentSetting(picIDs) # 多余，因为qwen2vl的默认解码策略是贪心解码，所以不需要画这个图
    changePicIDs = [handleLogDir(dirName) for dirName in qwenvl2changePicIDDirs]
    plotFigureOfComparePicIDChange(picIDs, changePicIDs, "qwen2VL")
    
    qwenvl25OriginalDirs = ["./eval/eval-2025-02-18-23-24", "./eval/eval-2025-02-19-16-01", "./eval/eval-2025-02-19-15-59", "./eval/eval-2025-02-20-03-24"]
    qwenvl25ChangeDirs = ["./eval/eval-2025-02-19-17-22", "./eval/eval-2025-02-20-02-02", "./eval/eval-2025-02-20-02-05", "./eval/eval-2025-02-20-13-07"]
    
    picIDs = [handleLogDir(dirName) for dirName in qwenvl25OriginalDirs]
    with open('./data/picSelect/self_Inference_Qwen2.5-VL-7B-Instruct.json', 'w') as f:
        json.dump(picIDs, f, ensure_ascii=False, indent=4) 
    changePicIds = [handleLogDir(dirName) for dirName in qwenvl25ChangeDirs]
    plotFigureOfComparePicIDChange(picIDs, changePicIds, "qwen25VL")
    
    qwen2VL_max_files = [ "./data/picSelect/qwen2VL_en_zh_en_Max_TransMetric_clipID2PicID.json", "./data/picSelect/qwen2VL_zh_en_en_Max_TransMetric_clipID2PicID.json" ]
    qwen25VL_max_files = [ "./data/picSelect/qwen25VL_en_zh_en_Max_TransMetric_clipID2PicID.json", "./data/picSelect/qwen25VL_zh_en_en_Max_TransMetric_clipID2PicID.json"]
    
    plotFigureOfMaxIndexDistribution(qwen2VL_max_files)
    plotFigureOfMaxIndexDistribution(qwen25VL_max_files)
    
    plotWhetherCometBLEUSame(qwen2VL_max_files)
    plotWhetherCometBLEUSame(qwen25VL_max_files)