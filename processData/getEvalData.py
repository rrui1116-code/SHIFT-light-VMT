import json

def getEvalData(testPath, clusteredPath):
    with open(testPath, "r") as f:
        testData = json.load(f)

    with open(clusteredPath, "r") as f:
        clusteredData = json.load(f)
    
    evalData = []
    for testDataItem, clusteredDataItem in zip(testData, clusteredData):
        
        if "language" in testDataItem:
            sourceLanguage = testDataItem["language"].split(":")[0].strip().upper()
        else:
            sourceLanguage = "EN"
        
        filtered_clipInfo = {
            "videoClipID": f"{testDataItem['video_id']}_{testDataItem['clip_id']}",
            "sentence": testDataItem[f'{sourceLanguage}_sentence'],
            "clusteredInfo": list(clusteredDataItem.values())[0]
        }
        evalData.append(filtered_clipInfo)
    
    return evalData

if __name__ == "__main__":
    enEvalData = getEvalData("./data/test/Test_general_en.json", "./data/frames/test_general_en_clustered.json")
    zhEvalData = getEvalData("./data/test/Test_general_zh.json", "./data/frames/test_general_zh_clustered.json")
    ambiguityEvalData = getEvalData("./data/test/Test_ambiguity.json", "./data/frames/test_ambiguity_clustered.json")

    with open("./data/frames/test_general_en_selector.json", "w") as f:
        json.dump(enEvalData, f, ensure_ascii=False, indent=4)

    with open("./data/frames/test_general_zh_selector.json", "w") as f:
        json.dump(zhEvalData, f, ensure_ascii=False, indent=4)

    with open("./data/frames/test_ambiguity_selector.json", "w") as f:
        json.dump(ambiguityEvalData, f, ensure_ascii=False, indent=4)


