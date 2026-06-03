from model.selector import selector, selectorConfig
from transformers import AutoProcessor, get_scheduler
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from LLM_vmt_dataset.vmtDataset import Dataset4SelectorTrain
from torch.utils.data import random_split
import torch
import torch.optim as optim
import argparse
import os
import logging
from datetime import datetime
from qwen_vl_utils import process_vision_info
import numpy as np
import sys
# 设置环境变量禁用tokenizers并行性，避免死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger('trainSelector')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

class SelectorDataCollator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device
        
    def __call__(self, batch):
        batchInputs = []
        labels = []
        
        for item in batch:  # batch是列表，每个item是字典
            videoClipID = item["videoClipID"]
            sentence = item["sentence"]
            
            # 添加图像-文本对
            for j in range(5):
                batchInputs.append([
                    {"role": "system", "content": ""},
                    {"role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"/root/autodl-tmp/frames/50frames/{videoClipID}/{videoClipID}_{int(item['clusteredInfo'][j])}.jpg",
                            "max_pixels": 360 * 420,
                        },
                    {"type": "text", "text": sentence},],
                    }
                ])
            
            # 添加纯文本
            batchInputs.append([
                {"role": "system", "content": ""},
                {"role": "user",
                "content": [
                    {"type": "text", "text": sentence},
                ],
            }])
            
            # 将原始分数通过幂函数变换放大差距
            original_scores = item["cometScores"]
            # 使用平方操作放大差距，保持[0,1]范围不变，增加分数间的差异
            amplified_scores = [score**2 for score in original_scores]
            labels.append(amplified_scores)
        
        imageInputs, videoInputs = process_vision_info(batchInputs)
        texts = [self.processor.apply_chat_template(itemInputs, tokenize=False, add_generation_prompt=False) for itemInputs in batchInputs]
        
        inputs = self.processor(
            text=texts,
            images=imageInputs,
            videos=videoInputs,
            padding=True,
            return_tensors="pt",
        )

        inputs["labels"] = torch.tensor(labels, dtype=torch.bfloat16)
        return inputs

# 自定义Trainer类
class SelectorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """
        重写log方法
        """
        # 调用父类的log方法
        super().log(logs, start_time)
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写evaluate方法
        """
        # 调用父类的evaluate方法
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        return metrics

def train(args):
    """
    训练模型
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 准备训练和验证数据集
    dataset = Dataset4SelectorTrain(args.train_data_path)

    if args.val_data_path is not None:
        # 使用独立验证集
        train_dataset = dataset
        val_dataset = Dataset4SelectorTrain(args.val_data_path)
    else:
        # 从训练集中按比例切分
        datasetGenerator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [1-args.val_ratio, args.val_ratio], generator=datasetGenerator)
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 加载模型和处理器
    logger.info(f"从 {args.model_path} 加载模型和处理器")
    
    config = selectorConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pairNum=6  # 设置为6个图文对
    )
    
    model = selector.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # 创建数据整理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_collator = SelectorDataCollator(processor, device)
    
    # 使用内联定义的配置
    training_args_dict = {
        "output_dir": f"{args.output_dir}/model",
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_strategy": 'steps',
        "logging_steps": args.logging_steps,
        "logging_dir": f"{args.output_dir}/tensorboard/",
        "eval_strategy": 'steps',
        "eval_steps": args.eval_steps,
        "save_strategy": 'steps',
        "save_steps": args.save_steps,
        "save_total_limit": 30,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": True,
        "report_to": ["tensorboard"],
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 2,
        "remove_unused_columns": False,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    # 初始化自定义Trainer
    trainer = SelectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.0
            ),
            # TensorBoard 集成通过 TrainingArguments 的 report_to 参数已启用
        ],
    )
    
    # 模型信息
    logger.info(model)
    logger.info(model.config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\n总参数量: {total_params:,} \n')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'\n可训练参数量: {total_trainable_params:,}\n')
    logger.info(training_args)
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 评估训练完成的模型
    logger.info("训练完成，进行最终评估...")
    eval_results = trainer.evaluate()
    logger.info(f"最终评估结果: {eval_results}")
    
    # 保存最终模型
    trainer.save_model(f"{args.output_dir}/best_model")
    processor.save_pretrained(f"{args.output_dir}/best_model")
    logger.info(f"训练完成，保存最终模型到 {args.output_dir}/best_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 selector 模型")
    
    # 数据集相关参数
    parser.add_argument("--train_data_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default=None, help="验证数据路径（可选，不提供则从训练集按比例切分）")
    parser.add_argument("--val_ratio", type=float, default=0.04, help="验证集比例（仅在未指定val_data_path时生效）")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="./checkpoint/selectorInit", help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default=None, help="模型保存路径")
    
    # 训练相关参数
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=600, help="评估步数")
    parser.add_argument("--save_steps", type=int, default=600, help="保存步数")
    
    # 断点继续训练相关参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点继续训练的路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output_dir is None:
        args.output_dir = f"./checkpoint/selectorTrained{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # log 设置
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, 'train.log'))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # 添加标准输出和标准错误重定向
    class TeeOutput:
        def __init__(self, file, original_stream):
            self.file = file
            self.original_stream = original_stream
        
        def write(self, data):
            self.file.write(data)
            self.original_stream.write(data)
            self.file.flush()
            self.original_stream.flush()
        
        def flush(self):
            self.file.flush()
            self.original_stream.flush()
    
    # 重定向标准输出和标准错误到文件
    output_log_file = open(os.path.join(args.output_dir, 'train_output.log'), 'a', encoding='utf-8')
    sys.stdout = TeeOutput(output_log_file, sys.stdout)
    sys.stderr = TeeOutput(output_log_file, sys.stderr)
    
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # 开始训练
    train(args) 