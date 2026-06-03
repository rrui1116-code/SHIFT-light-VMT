import os
import shutil
import argparse
import json



def mvDataToAnalysis(clipID):
    os.makedirs(f"./temp/analysis/{clipID}", exist_ok=True)
    videoID = clipID[:11]
    
    try:
        for i in range(10):
            sourcePath = f"./data/frames/{videoID}/{clipID}_{i}.jpg"
            destinationPath = f"./temp/analysis/{clipID}/{clipID}_{i}.jpg"
            shutil.copy(sourcePath, destinationPath)
        
        sourcePath = f"./data/trainVideoClips/{videoID}/{clipID}.mp4"
        destinationPath = f"./temp/analysis/{clipID}/{clipID}.mp4"
        shutil.copy(sourcePath, destinationPath)
        
    except FileNotFoundError:
        print(f"源文件 {sourcePath} 不存在")
    except PermissionError:
        print(f"没有权限写入目标路径 {destinationPath}")
    except Exception as e:
        print(f"发生错误: {e}")
        

def getRelatedSentences(clipID, modelName, transDriection):
    with open(f"./data/picSelect/{modelName}_{transDriection}_en_Max_TransMetric_clipID2PicID.json", "r") as f:
        data = json.load(f)
    for item in data:
        if item["clipID"] == clipID:
            with open(f"./temp/analysis/{clipID}/relatedSentences.json", "w") as f:
                json.dump([item], f, ensure_ascii=False, indent=2)
        break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--clipID", type=str, required=True, default=None, help="Clip ID need to be moved")
    parser.add_argument("--model_name", type=str, default="qwen25VL", help="Model name")
    parser.add_argument("--en_zh", action="store_false", help="Default is Chinese to English, if set, English to Chinese")
    args = parser.parse_args()
    
    transDriection = "zh_en" if args.en_zh else "en_zh" 
    
    mvDataToAnalysis(args.clipID)
    getRelatedSentences(args.clipID, args.model_name, transDriection)
