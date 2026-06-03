"""
    该文件用于将视频均匀提取指定数量的帧和中间帧，并进行保存为jpg。
"""
import argparse
import os
import logging
import json
import cv2
import gc
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 日志设置
logger = logging.getLogger('extractFrames')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')

def save_frame(output_path, frame):
    """保存单张图片"""
    cv2.imwrite(output_path, frame)

def extract_frames(clipID, num_frames=10, extract_middle_frame=True, vatex=False):
    """
    从视频中提取均匀间隔的 num_frames 帧（以及可选的中间帧），
    并利用线程池并行保存图片，缩短 I/O 阻塞时间。
    """
    try:
        if vatex:
            output_dir = f'/root/autodl-tmp/frames/{num_frames}frames/{clipID}'
            videoID = clipID
        else:
            videoID = clipID[:11]
            output_dir = f'/root/autodl-tmp/frames/{num_frames}frames/{videoID}'
        # 如果最后一帧已存在，认为该视频已处理
        check_path = os.path.join(output_dir, f'{clipID}_{num_frames-1}.jpg')
        if os.path.exists(check_path) and os.path.getsize(check_path) > 0:
            return True

        os.makedirs(output_dir, exist_ok=True)
        if vatex:
            video_path = f"/root/autodl-tmp/videos_small/{clipID}.mp4"
        else:
            video_path = f"/root/autodl-tmp/videos_small/{videoID}/{clipID}.mp4"
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error(f"{clipID} 无有效帧数，跳过处理。")
            cap.release()
            return False

        interval = total_frames // num_frames
        middle_frame_index = total_frames // 2

        extracted_frames = []
        error_count = 0
        # 提取均匀间隔的帧
        for i in range(num_frames):
            frame_index = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                extracted_frames.append(frame)
            else:
                logger.error(f"{clipID} 在帧 {frame_index} 读取失败。")
                extracted_frames.append(None)
                error_count += 1

        # 如果需要，则提取中间帧
        if extract_middle_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, middle_frame = cap.read()
            if ret:
                extracted_frames.append(middle_frame)
            else:
                logger.error(f"{clipID} 在中间帧 {middle_frame_index} 读取失败。")
                error_count += 1

        cap.release()
        gc.collect()

        # 检测是否出现严重的解码错误
        if error_count > 0:
            logger.warning(f"H264解码错误ID: {clipID} (发现{error_count}处读取失败)")

        # 并行保存图片，减少单线程 I/O 开销
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 保存均匀提取的帧
            for i, frame in enumerate(extracted_frames[:num_frames]):
                if frame is not None:
                    out_path = os.path.join(output_dir, f'{clipID}_{i}.jpg')
                    tasks.append(executor.submit(save_frame, out_path, frame))
            # 保存中间帧（如果提取且读取成功）
            if extract_middle_frame and len(extracted_frames) > num_frames and extracted_frames[-1] is not None:
                mid_path = os.path.join(output_dir, f'{clipID}_mid.jpg')
                tasks.append(executor.submit(save_frame, mid_path, extracted_frames[-1]))
            # 等待所有保存任务完成
            for task in tasks:
                task.result()

        return True

    except Exception as e:
        logger.error(f"{clipID} 提取失败! {e}")
        return False

if __name__ == '__main__':
    
    # log 设置
    os.makedirs('./log', exist_ok=True)
    fileHandler = logging.FileHandler('./log/extractFrames.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--json_file_path", type=str, required=True, help="The json file path of video clips")
    parser.add_argument('-n', "--num_frames", type=int, default=10, help="Number of Sampled frames")
    parser.add_argument('-m', "--extract_middle_frame", action='store_true', help="Whether to extract middle frame")
    parser.add_argument('-v', "--vatex", action='store_true', help="Whether to extract vatex data")
    args = parser.parse_args()

    # 根据 num_frames 创建目标文件夹
    
    if args.vatex:
        os.makedirs(f'/root/autodl-tmp/frames/{args.num_frames}frames', exist_ok=True)
    else:
        os.makedirs(f'/root/autodl-tmp/frames/{args.num_frames}frames', exist_ok=True)

    # 读取 JSON 数据并构造 clipID 列表
    with open(args.json_file_path, 'r') as f:
        all_data = json.load(f)
        
    if args.vatex:
        clip_ids = [f'{item["videoID"]}' for item in all_data]
    else:
        clip_ids = [f'{item["video_id"]}_{item["clip_id"]}' for item in all_data]

    # 使用动态进程数
    pool_size = 8
    specific_extract_frames = partial(extract_frames, num_frames=args.num_frames, extract_middle_frame=args.extract_middle_frame, vatex=args.vatex)
    
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(specific_extract_frames, clip_ids),
                            total=len(clip_ids),
                            desc="Processing Data"))
    
    success_count = sum(1 for r in results if r)
    logger.info(f"{success_count}/{len(clip_ids)} 个视频片段提取成功!")
