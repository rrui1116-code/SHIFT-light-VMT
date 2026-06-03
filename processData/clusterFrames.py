"""
    之前使用extractFrames.py为每个视频片段均匀提取出50帧画面。
    再使用这个程序对每个视频片段的50帧画面进行聚类，提取出5个代表帧，并保存。
    代表帧的选择标准是：清晰度最高的帧。
"""

import json
import argparse
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class framesExtractedDataset(Dataset):
    def __init__(self, jsonFilePath):
        with open(jsonFilePath, 'r') as f:
            allData = json.load(f)
        self.allClipID = [f'{i["videoID"]}' for i in allData]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.allClipID)
    
    def _calculate_sharpness(self, frame):
        """使用Laplacian Variance法评估清晰度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def __getitem__(self, idx):
        clipID = self.allClipID[idx]
        imagePaths = [f"/root/autodl-tmp/frames/50frames/{clipID}/{clipID}_{i}.jpg" for i in range(50)]
        images, sharpness = [], []
        for path in imagePaths:
            # 只读取一次图像
            frame = cv2.imread(path)
            if frame is None:
                continue
            # 计算清晰度
            sharp = self._calculate_sharpness(frame)
            sharpness.append(sharp)
            # 将 BGR 转为 RGB，并转换为 PIL Image，再应用预处理
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            images.append(image)
        
        # 简单补齐策略：如果读取帧数不足50，则用最后一帧补齐
        if len(images) != 50:
            last_image = images[-1] if images else torch.zeros(3, 224, 224)
            last_sharp = sharpness[-1] if sharpness else 0.0
            while len(images) < 50:
                images.append(last_image)
                sharpness.append(last_sharp)
        
        return clipID, torch.stack(images), torch.tensor(sharpness)

def cluster_frames(features, K=5):
    """使用KMeans聚类帧"""
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(features)
    return labels

def process_videos(dataLoader, model):
    selectedIDs = []
    
    for clipIDs, images, sharpness in tqdm(dataLoader):
        # images 形状：(batch, 50, C, H, W)
        batch_size, num_frames, C, H, W = images.shape
        # 将批次内所有帧展平成 (batch * 50, C, H, W)
        images = images.view(-1, C, H, W).to("cuda", non_blocking=True)
        
        with torch.no_grad():
            features = model(images)
        
        # 恢复为 (batch, 50, feature_dim)
        features = features.view(batch_size, num_frames, -1).cpu().numpy()
        sharpness = sharpness.numpy()
        
        # 顺序处理每个视频片段的聚类
        clusterLabels = [cluster_frames(feature) for feature in features]
        
        for clipID, clusterLabel, sharpnessScore in zip(clipIDs, clusterLabels, sharpness):
            unique_classes = np.unique(clusterLabel)
            bestIndices = []
            for c in unique_classes:
                indices = np.where(clusterLabel == c)[0]
                # 选择该类别下清晰度最高的帧的索引
                best_index = indices[np.argmax(sharpnessScore[clusterLabel == c])]
                bestIndices.append(int(best_index))
            # 若聚类类别不足5类，则补充最后一类的结果
            if len(bestIndices) < 5:
                bestIndices += [bestIndices[-1]] * (5 - len(bestIndices))
            selectedIDs.append( {clipID: bestIndices} )

    return selectedIDs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--json_file_path", type=str, required=True)
    parser.add_argument('-o', "--output_file_path", type=str, required=True)
    args = parser.parse_args()
    
    # 初始化 ResNet50 模型并去掉最后全连接层，用于特征提取
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to("cuda")
    model.fc = nn.Identity() 
    model.eval()
    
    dataset = framesExtractedDataset(args.json_file_path)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, 
                                pin_memory=True, persistent_workers=True, prefetch_factor=4)
    selectedIDs = process_videos(dataLoader, model)
    
    with open(args.output_file_path, 'w') as f:
        json.dump(selectedIDs, f, ensure_ascii=False, indent=4)