import os
import json
import shutil
from processData.extractFrames import extract_frames

def checkAllVideoClipsAvail(srcVideoClipsDir):
    videoIDs = set(os.listdir(srcVideoClipsDir))
    with open('./data/test/Test_ambiguity.json', 'r') as f:
        data = json.load(f)
        for clip in data:
            if clip['video_id'] not in videoIDs:
                raise ValueError(f'{clip["video_id"]} not found!')
    
    for videoID in videoIDs:
        if os.path.exists(f'./data/trainVideoClips/{videoID}'):
            continue
        shutil.copytree(f'{srcVideoClipsDir}/{videoID}', f'./data/trainVideoClips/{videoID}')
    
    for clip in data:
        extract_frames(f"{clip['video_id']}_{clip['clip_id']}")


if __name__ == '__main__':
    
    checkAllVideoClipsAvail('./temp/testVideos/trainVideoClips')