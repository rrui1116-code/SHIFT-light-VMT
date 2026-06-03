import os
import json
import logging
import numpy as np
from utils.computeTransMetric import getSrcPredsRefs, computeCOMET, computeBLEU, computeBLEURT

logger = logging.getLogger('getMetric')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def getMaxIndices(Dirs, TransDirection, modelName):
    srcPredsRefs = [ getSrcPredsRefs(DirName) for DirName in Dirs ]
    
    with open(os.path.join(Dirs[0], "results.json"), 'r') as f:
        prdsData = json.load(f)
    outputResults = [ {"clipID": item["clipID"]} for item in prdsData]
    
    cometScores = np.array([ computeCOMET(src, preds, refs)["scores"] for src, preds, refs in srcPredsRefs ]) * 100 
    max_COMET_score = np.mean(np.max(cometScores, axis=0))
    logger.info(f"\033[91m Max COMET score for {modelName} {TransDirection}: {max_COMET_score} \033[0m")
    max_COMET_indices = np.argmax(cometScores, axis=0)
    # 将 max_indices 转换为 Python 列表（numpy 数组不是 JSON 可序列化的）
    max_COMET_indices = max_COMET_indices.tolist()
    
    # 因为BLEU值计算时单个计算取平均和多个计算取平均的结果是不一样的，所以这里要分开计算
    BLEUScores = []
    for src, preds, refs in srcPredsRefs:
        BLEUScores.append([computeBLEU([pred], [ref], TransDirection[-2:]=='zh', True) for pred, ref in zip(preds, refs)])
    BLEUScores = np.array(BLEUScores)
    max_BLEU_indices = np.argmax(BLEUScores, axis=0)
    max_BLEU_indices = max_BLEU_indices.tolist()
    MaxPreds = [srcPredsRefs[maxIndex][1][i] for i, maxIndex in enumerate(max_BLEU_indices)]
    max_BLEU_score = computeBLEU(MaxPreds, srcPredsRefs[0][2], TransDirection[-2:]=='zh', True)
    logger.info(f"\033[91m Max BLEU score for {modelName} {TransDirection}: {max_BLEU_score} \033[0m")
    
    BLEURTScores = np.array([computeBLEURT(pred, ref, returnAverage=False) for src, pred, ref in srcPredsRefs]) * 100
    max_BLEURT_score = np.mean(np.max(BLEURTScores, axis=0))
    logger.info(f"\033[91m Max BLEURT score for {modelName} {TransDirection}: {max_BLEURT_score} \033[0m")
    max_BLEURT_indices = np.argmax(BLEURTScores, axis=0).tolist()
    
    assert len(max_COMET_indices) == len(max_BLEU_indices) == len(outputResults) == len(max_BLEURT_indices), \
        "Lengths of max indices should be the same"
    
    for i in range(len(max_BLEU_indices)):
        outputResults[i]["max_COMET_picID"] = max_COMET_indices[i]
        outputResults[i]["max_BLEU_picID"] = max_BLEU_indices[i]
        outputResults[i]["max_BLEURT_picID"] = max_BLEURT_indices[i]
        outputResults[i]["max_COMET_score"] = float(cometScores[max_COMET_indices[i], i])
        outputResults[i]["COMET_scores"] = cometScores[:, i].tolist()
        outputResults[i]["max_BLEU_score"] = float(BLEUScores[max_BLEU_indices[i], i])
        outputResults[i]["BLEU_scores"] = BLEUScores[:, i].tolist()
        outputResults[i]["max_BLEURT_score"] = float(BLEURTScores[max_BLEURT_indices[i], i])
        outputResults[i]["BLEURT_scores"] = BLEURTScores[:, i].tolist()
        outputResults[i]["src_sentence"] = srcPredsRefs[0][0][i]
        outputResults[i]["tgt_sentence"] = srcPredsRefs[0][2][i]
        outputResults[i]["pred_sentences"] = [srcPredsRefs[j][1][i] for j in range(len(srcPredsRefs))]

    with open(f"./data/picSelect/{modelName}_{TransDirection}_en_Max_TransMetric_clipID2PicID.json", "w") as f:
        json.dump(outputResults, f, ensure_ascii=False, indent=2)
    
    return max_COMET_indices, max_BLEU_indices
    
# 使用最高的COMET值的下标计算BLEU值，使用最高的BLEU值的下标计算COMET值，查看分数
def exchangeIndex(Dirs, TransDirection, max_COMET_indices, max_BLEU_indices, modelName):
    srcPredsRefs = [ getSrcPredsRefs(DirName) for DirName in Dirs ]
    srcs, refs = srcPredsRefs[0][0], srcPredsRefs[0][2]
    maxBLEUPreds = [srcPredsRefs[maxIndex][1][i] for i, maxIndex in enumerate(max_BLEU_indices)]
    maxCOMETPreds = [srcPredsRefs[maxIndex][1][i] for i, maxIndex in enumerate(max_COMET_indices)]
    
    COMETScoreByMaxBLEU = computeCOMET(srcs, maxBLEUPreds, refs)["mean_score"] * 100
    logger.info(f"\033[91m COMET score for {modelName} {TransDirection} by max BLEU indices: {COMETScoreByMaxBLEU} \033[0m")
    
    BLEUScoreByMaxCOMET = computeBLEU(maxCOMETPreds, refs, TransDirection[-2:]=='zh', True)
    logger.info(f"\033[91m BLEU score for {modelName} {TransDirection} by max COMET indices: {BLEUScoreByMaxCOMET} \033[0m")
    BLEURTScoreByMaxCOMET = computeBLEURT(maxCOMETPreds, refs) * 100
    logger.info(f"\033[91m BLEURT score for {modelName} {TransDirection} by max COMET indices: {BLEURTScoreByMaxCOMET} \033[0m")
    
def readFileGetIndices(filePath):
    with open(filePath, "r") as f:
        data = json.load(f)
    return [item["max_COMET_picID"] for item in data], [item["max_BLEU_picID"] for item in data], 

if __name__ == "__main__":
    
    fileHandler = logging.FileHandler(f'./log/MaxScore.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    qwen25VL_en_zh_Dirs = ["./eval/eval-2025-02-24-16-58", "./eval/eval-2025-02-24-19-44", "./eval/eval-2025-02-24-20-43", "./eval/eval-2025-02-24-21-43", "./eval/eval-2025-02-24-22-53", \
        "./eval/eval-2025-02-24-23-42", "./eval/eval-2025-02-25-00-42", "./eval/eval-2025-02-25-01-41", "./eval/eval-2025-02-25-02-41", "./eval/eval-2025-02-25-03-41"]
    qwen25VL_zh_en_Dirs = ["./eval/eval-2025-02-24-16-55", "./eval/eval-2025-02-24-19-46", "./eval/eval-2025-02-24-20-45", "./eval/eval-2025-02-24-21-44", "./eval/eval-2025-02-24-23-53", \
        "./eval/eval-2025-02-24-23-48", "./eval/eval-2025-02-25-00-47", "./eval/eval-2025-02-25-01-46", "./eval/eval-2025-02-25-02-45", "./eval/eval-2025-02-25-03-44"]
    
    qwen2VL_en_zh_Dirs = [ "./eval/eval-2025-02-25-22-37", "./eval/eval-2025-02-25-23-32", "./eval/eval-2025-02-26-00-27", "./eval/eval-2025-02-26-01-21", "./eval/eval-2025-02-26-02-16", \
        "./eval/eval-2025-02-26-03-10", "./eval/eval-2025-02-26-04-04", "./eval/eval-2025-02-26-04-59", "./eval/eval-2025-02-26-05-53", "./eval/eval-2025-02-26-06-47" ]
    qwen2VL_zh_en_Dirs = [ "./eval/eval-2025-02-25-22-36", "./eval/eval-2025-02-25-23-33", "./eval/eval-2025-02-26-00-26", "./eval/eval-2025-02-26-01-19", "./eval/eval-2025-02-26-02-12", \
        "./eval/eval-2025-02-26-03-06", "./eval/eval-2025-02-26-03-58", "./eval/eval-2025-02-26-04-52", "./eval/eval-2025-02-26-05-45", "./eval/eval-2025-02-26-06-38" ]
    
    qwen2VL_ambiguity_Dirs = ['./eval/eval-2025-03-11-19-32', './eval/eval-2025-03-11-19-52', './eval/eval-2025-03-11-20-00', './eval/eval-2025-03-11-20-08', \
        './eval/eval-2025-03-11-20-17', './eval/eval-2025-03-11-20-25', './eval/eval-2025-03-11-20-33', './eval/eval-2025-03-11-20-41', './eval/eval-2025-03-11-20-49', './eval/eval-2025-03-11-20-57']
    qwen25VL_ambiguity_Dirs = ['./eval/eval-2025-03-11-19-36', './eval/eval-2025-03-11-19-54', './eval/eval-2025-03-11-20-02', './eval/eval-2025-03-11-20-11', \
        './eval/eval-2025-03-11-20-19', './eval/eval-2025-03-11-20-28', './eval/eval-2025-03-11-20-36', './eval/eval-2025-03-11-20-45', './eval/eval-2025-03-11-20-53', './eval/eval-2025-03-11-21-02']

    qwen2VL_cluster_en_zh_Dirs = ['./eval/eval-2025-03-14-16-29', './eval/eval-2025-03-14-17-23', './eval/eval-2025-03-14-18-18', './eval/eval-2025-03-14-19-12', './eval/eval-2025-03-14-20-06']
    qwen2VL_cluster_zh_en_Dirs = ['./eval/eval-2025-03-14-16-31', './eval/eval-2025-03-14-17-24', './eval/eval-2025-03-14-18-17', './eval/eval-2025-03-14-19-10', './eval/eval-2025-03-14-20-03']
    qwen2VL_cluster_ambiguity_Dirs = ['./eval/eval-2025-03-16-16-48', './eval/eval-2025-03-16-16-56', './eval/eval-2025-03-16-17-04', './eval/eval-2025-03-16-17-11', './eval/eval-2025-03-16-17-19']
    
    qwen25VL_cluster_en_zh_Dirs = ['./eval/eval-2025-03-14-21-01', './eval/eval-2025-03-14-22-01', './eval/eval-2025-03-14-23-00', './eval/eval-2025-03-15-00-00', './eval/eval-2025-03-15-00-59', './eval/eval-2025-03-11-23-57']
    qwen25VL_cluster_zh_en_Dirs = ['./eval/eval-2025-03-14-20-56', './eval/eval-2025-03-14-21-55', './eval/eval-2025-03-14-22-54', './eval/eval-2025-03-14-23-53', './eval/eval-2025-03-15-00-53', './eval/eval-2025-03-11-22-44']
    qwen25VL_cluster_ambiguity_Dirs = ['./eval/eval-2025-03-16-17-26', './eval/eval-2025-03-16-17-35', './eval/eval-2025-03-16-17-43', './eval/eval-2025-03-16-17-52', './eval/eval-2025-03-16-18-00', './eval/eval-2025-03-25-15-25']
    
    qwen25VL_en_zh_max_COMET_indices, qwen25VL_en_zh_max_BLEU_indices = getMaxIndices(qwen25VL_en_zh_Dirs, "en_zh", "qwen25VL")
    qwen25VL_zh_en_max_COMET_indices, qwen25VL_zh_en_max_BLEU_indices = getMaxIndices(qwen25VL_zh_en_Dirs, "zh_en", "qwen25VL")
    
    qwen2VL_en_zh_max_COMET_indices, qwen2VL_en_zh_max_BLEU_indices = getMaxIndices(qwen2VL_en_zh_Dirs, "en_zh", "qwen2VL")
    qwen2VL_zh_en_max_COMET_indices, qwen2VL_zh_en_max_BLEU_indices = getMaxIndices(qwen2VL_zh_en_Dirs, "zh_en", "qwen2VL")
    
    qwen25VL_cluster_en_zh_max_COMET_indices, qwen25VL_cluster_en_zh_max_BLEU_indices = getMaxIndices(qwen25VL_cluster_en_zh_Dirs, "en_zh", "qwen25VL_cluster")
    qwen25VL_cluster_zh_en_max_COMET_indices, qwen25VL_cluster_zh_en_max_BLEU_indices = getMaxIndices(qwen25VL_cluster_zh_en_Dirs, "zh_en", "qwen25VL_cluster")
    
    qwen2VL_cluster_en_zh_max_COMET_indices, qwen2VL_cluster_en_zh_max_BLEU_indices = getMaxIndices(qwen2VL_cluster_en_zh_Dirs, "en_zh", "qwen2VL_cluster")
    qwen2VL_cluster_zh_en_max_COMET_indices, qwen2VL_cluster_zh_en_max_BLEU_indices = getMaxIndices(qwen2VL_cluster_zh_en_Dirs, "zh_en", "qwen2VL_cluster")
    
    qwen2VL_ambiguity_max_COMET_indices, qwen2VL_ambiguity_max_BLEU_indices = getMaxIndices(qwen2VL_ambiguity_Dirs, "en_zh", "qwen2VL_ambiguity")
    qwen25VL_ambiguity_max_COMET_indices, qwen25VL_ambiguity_max_BLEU_indices = getMaxIndices(qwen25VL_ambiguity_Dirs, "en_zh", "qwen25VL_ambiguity")

    qwen2VL_cluster_ambiguity_max_COMET_indices, qwen2VL_cluster_ambiguity_max_BLEU_indices = getMaxIndices(qwen2VL_cluster_ambiguity_Dirs, "en_zh", "qwen2VL_cluster_ambiguity")
    qwen25VL_cluster_ambiguity_max_COMET_indices, qwen25VL_cluster_ambiguity_max_BLEU_indices = getMaxIndices(qwen25VL_cluster_ambiguity_Dirs, "en_zh", "qwen25VL_cluster_ambiguity")

    qwen25VL_en_zh_max_COMET_indices, qwen25VL_en_zh_max_BLEU_indices = readFileGetIndices("./data/picSelect/qwen25VL_en_zh_en_Max_TransMetric_clipID2PicID.json")
    qwen25VL_zh_en_max_COMET_indices, qwen25VL_zh_en_max_BLEU_indices = readFileGetIndices("./data/picSelect/qwen25VL_zh_en_en_Max_TransMetric_clipID2PicID.json")
    
    qwen2VL_en_zh_max_COMET_indices, qwen2VL_en_zh_max_BLEU_indices = readFileGetIndices("./data/picSelect/qwen2VL_en_zh_en_Max_TransMetric_clipID2PicID.json")
    qwen2VL_zh_en_max_COMET_indices, qwen2VL_zh_en_max_BLEU_indices = readFileGetIndices("./data/picSelect/qwen2VL_zh_en_en_Max_TransMetric_clipID2PicID.json")
    
    exchangeIndex(qwen25VL_en_zh_Dirs, "en_zh", qwen25VL_en_zh_max_COMET_indices, qwen25VL_en_zh_max_BLEU_indices, "qwen25VL")
    exchangeIndex(qwen25VL_zh_en_Dirs, "zh_en", qwen25VL_zh_en_max_COMET_indices, qwen25VL_zh_en_max_BLEU_indices, "qwen25VL")
    
    exchangeIndex(qwen2VL_en_zh_Dirs, "en_zh", qwen2VL_en_zh_max_COMET_indices, qwen2VL_en_zh_max_BLEU_indices, "qwen2VL")
    exchangeIndex(qwen2VL_zh_en_Dirs, "zh_en", qwen2VL_zh_en_max_COMET_indices, qwen2VL_zh_en_max_BLEU_indices, "qwen2VL")
    
    exchangeIndex(qwen25VL_cluster_en_zh_Dirs, "en_zh", qwen25VL_cluster_en_zh_max_COMET_indices, qwen25VL_cluster_en_zh_max_BLEU_indices, "qwen25VL_cluster")
    exchangeIndex(qwen25VL_cluster_zh_en_Dirs, "zh_en", qwen25VL_cluster_zh_en_max_COMET_indices, qwen25VL_cluster_zh_en_max_BLEU_indices, "qwen25VL_cluster")
    
    exchangeIndex(qwen2VL_cluster_en_zh_Dirs, "en_zh", qwen2VL_cluster_en_zh_max_COMET_indices, qwen2VL_cluster_en_zh_max_BLEU_indices, "qwen2VL_cluster")
    exchangeIndex(qwen2VL_cluster_zh_en_Dirs, "zh_en", qwen2VL_cluster_zh_en_max_COMET_indices, qwen2VL_cluster_zh_en_max_BLEU_indices, "qwen2VL_cluster")
    
    exchangeIndex(qwen2VL_ambiguity_Dirs, "en_zh", qwen2VL_ambiguity_max_COMET_indices, qwen2VL_ambiguity_max_BLEU_indices, "qwen2VL_ambiguity")
    exchangeIndex(qwen25VL_ambiguity_Dirs, "en_zh", qwen25VL_ambiguity_max_COMET_indices, qwen25VL_ambiguity_max_BLEU_indices, "qwen25VL_ambiguity")

    exchangeIndex(qwen2VL_cluster_ambiguity_Dirs, "en_zh", qwen2VL_cluster_ambiguity_max_COMET_indices, qwen2VL_cluster_ambiguity_max_BLEU_indices, "qwen2VL_cluster_ambiguity")
    exchangeIndex(qwen25VL_cluster_ambiguity_Dirs, "en_zh", qwen25VL_cluster_ambiguity_max_COMET_indices, qwen25VL_cluster_ambiguity_max_BLEU_indices, "qwen25VL_cluster_ambiguity")
