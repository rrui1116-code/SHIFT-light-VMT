import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union, Dict
import torch.nn.functional as F
from torch.nn import MSELoss
from transformers.activations import ACT2FN

# Assuming these imports exist based on the original code context
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLModel # Added Qwen2_5_VLModel here
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)

# 仿照Qwen2_5_VLForConditionalGeneration新增一个专门用于排序任务的输出类
@dataclass
class selectorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    accuracy: Optional[torch.FloatTensor] = None  # 添加准确率字段

# 仿照Qwen2_5_VLConfig新增一个专门用于排序任务的配置类
class selectorConfig(Qwen2_5_VLConfig):
    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling=None,
        numOfDecoderLayers=1,
        pairNum=6,
        lambdaOfLoss2=0.3,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            vision_config=vision_config,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        
        self.pairNum = pairNum
        self.lambdaOfLoss2 = lambdaOfLoss2
        self.numOfDecoderLayers = numOfDecoderLayers

class selector(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: selectorConfig):
        super().__init__(config)
        # 删除lm_head
        if hasattr(self, 'lm_head'):
            del self.lm_head
            
        if hasattr(self.model, 'layers') and len(self.model.layers) > config.numOfDecoderLayers:
            # Keep only the first config.numOfDecoderLayers decoder layers
            self.model.layers = nn.ModuleList([self.model.layers[i] for i in range(config.numOfDecoderLayers)])
            # Optional: Update config for clarity, but might affect compatibility.
            # self.config.num_hidden_layers = config.numOfDecoderLayers
        elif not hasattr(self.model, 'layers'):
            logger.warning("The model structure does not have 'layers' attribute as expected.")
        elif len(self.model.layers) <= config.numOfDecoderLayers:
            logger.info(f"Model already has {config.numOfDecoderLayers} or fewer decoder layers. No pruning needed.")
        
        # 添加排序头部网络，用于输出图文对的分数
        hidden_size = config.hidden_size
        intermediate_size = hidden_size * 4  # 典型的FFN中间层扩展比例
        
        # 获取模型使用的激活函数
        act_fn = config.hidden_act if hasattr(config, "hidden_act") else "silu"
        self.activation_fn = ACT2FN[act_fn]
        
        # 使用标准FFN架构
        self.ranking_norm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)
        self.ranking_dense = nn.Linear(hidden_size, intermediate_size)
        self.ranking_dropout = nn.Dropout(0.1 if self.training else 0.0)
        self.ranking_out_proj = nn.Linear(intermediate_size, 1)  # 输出原始分数，不使用Sigmoid
        
        # 冻结除了self.model和排序头部相关参数之外的所有参数
        for name, param in self.named_parameters():
            if not any(module_name in name for module_name in ['model.', 'ranking_norm.', 'ranking_dense.', 'ranking_dropout.', 'ranking_out_proj.']):
                param.requires_grad = False
                
        # 打印参数状态
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = total_params - frozen_params
        logger.info(f"已冻结参数: {frozen_params:,} ({frozen_params/total_params:.2%})")
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练模型加载并初始化selector
        
        Args:
            pretrained_model_name_or_path: 预训练模型路径
            *model_args: 传递给父类from_pretrained的位置参数
            **kwargs: 传递给父类from_pretrained的关键字参数
            
        Returns:
            selector: 初始化后的selector模型实例
        """
        # 首先加载原始配置
        config_kwargs = kwargs.get("config_kwargs", {})
        config = selectorConfig.from_pretrained(
            pretrained_model_name_or_path, **config_kwargs
        )
        
        # 确保使用selectorConfig
        if "config" in kwargs:
            if not isinstance(kwargs["config"], selectorConfig):
                # 如果提供了配置但不是selectorConfig，则转换它
                provided_config = kwargs["config"]
                selector_config = selectorConfig(
                    vocab_size=provided_config.vocab_size,
                    hidden_size=provided_config.hidden_size,
                    intermediate_size=provided_config.intermediate_size,
                    num_hidden_layers=provided_config.num_hidden_layers,
                    num_attention_heads=provided_config.num_attention_heads,
                    num_key_value_heads=provided_config.num_key_value_heads,
                    hidden_act=provided_config.hidden_act,
                    max_position_embeddings=provided_config.max_position_embeddings,
                    initializer_range=provided_config.initializer_range,
                    rms_norm_eps=provided_config.rms_norm_eps,
                    use_cache=provided_config.use_cache,
                    tie_word_embeddings=provided_config.tie_word_embeddings,
                    rope_theta=provided_config.rope_theta,
                    vision_config=provided_config.vision_config,
                    rope_scaling=provided_config.rope_scaling,
                )
                kwargs["config"] = selector_config
        else:
            kwargs["config"] = config
        
        # 保存是否需要使用bf16
        use_bf16 = kwargs.pop("dtype", None) == torch.bfloat16
        
        # 调用父类的from_pretrained方法加载预训练模型
        model = super(selector, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        
        # 如果需要，转换为bf16
        if use_bf16:
            model = model.to_bf16()
        
        return model
    
    @staticmethod
    def listnet_loss(predictions, targets):
        """
        ListNet排序损失函数，基于概率分布的KL散度
        """
        # 将预测和目标转换为概率分布
        p_pred = F.softmax(predictions, dim=-1)
        p_target = F.softmax(targets, dim=-1)
        
        # 计算KL散度
        loss = -torch.sum(p_target * torch.log(p_pred + 1e-10), dim=-1)
        return loss.mean()
    
    @staticmethod
    def listmle_loss(predictions, targets):
        """
        ListMLE排序损失函数，基于似然最大化
        """
        # 获取序列长度
        batch_size, seq_len = predictions.size()
        
        # 按照目标得分排序的索引
        sorted_idx = torch.argsort(targets, dim=1, descending=True)
        
        # 按排序索引重新排列预测
        sorted_pred = torch.gather(predictions, 1, sorted_idx)
        
        # 计算ListMLE损失
        loss = 0
        for i in range(batch_size):
            curr_pred = sorted_pred[i]
            for j in range(seq_len):
                # 计算当前位置的似然 P(r_j | r_1, ..., r_{j-1})
                if j == seq_len - 1:
                    loss -= curr_pred[j]
                else:
                    loss -= curr_pred[j] - torch.log(torch.sum(torch.exp(curr_pred[j:])))
        
        return loss / batch_size
    
    @staticmethod
    def ranknet_loss(predictions, targets):
        """
        RankNet排序损失函数，基于配对比较
        适用于未经过sigmoid的原始分数
        """
        batch_size, seq_len = predictions.size()
        
        # 创建所有可能的配对
        pred_pairs = []
        target_pairs = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(j+1, seq_len):
                    pred_diff = predictions[i, j] - predictions[i, k]
                    target_diff = targets[i, j] - targets[i, k]
                    
                    # 只考虑目标分数不同的配对
                    if abs(target_diff) > 1e-5:  # 使用小阈值避免浮点误差
                        pred_pairs.append(pred_diff)
                        # 转换为二元标签：1表示j优于k，0表示k优于j
                        target_pairs.append(1.0 if target_diff > 0 else 0.0)
        
        if not pred_pairs:  # 如果没有有效的配对
            return torch.tensor(0.0, device=predictions.device)
        
        # 转换为张量
        pred_pairs = torch.stack(pred_pairs)
        target_pairs = torch.tensor(target_pairs, device=predictions.device, dtype=torch.float)
        
        # 计算二元交叉熵损失
        # 对于未经过sigmoid的分数，binary_cross_entropy_with_logits是适合的
        loss = F.binary_cross_entropy_with_logits(pred_pairs, target_pairs)
        return loss

    @staticmethod
    def calculate_accuracy(predictions, targets):
        """
        计算Top-1准确率 - 考虑到标签中可能存在多个最大值的情况
        
        Args:
            predictions: 形状为 [batch_size, num_pairs] 的预测分数
            targets: 形状为 [batch_size, num_pairs] 的目标分数
            
        Returns:
            accuracy: Top-1准确率
        """
        batch_size = predictions.size(0)
        correct = 0
        
        for i in range(batch_size):
            # 获取当前样本的预测和目标
            pred = predictions[i]
            target = targets[i]
            
            # 找出预测的最大值索引
            pred_max_idx = torch.argmax(pred).item()
            
            # 找出目标中所有最大值的索引
            max_target_value = torch.max(target)
            max_target_indices = (target == max_target_value).nonzero(as_tuple=True)[0]
            
            # 如果预测的最大值索引在目标最大值索引中，则认为预测正确
            if pred_max_idx in max_target_indices:
                correct += 1
        
        # 计算准确率
        accuracy = correct / batch_size
        return torch.tensor(accuracy, device=predictions.device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,  # 修改为浮点型，表示排序分数，0-1之间
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, selectorOutput]:  # 修改返回类型为RankingOutput
        # 修改forward函数，只保留一层decoder，小幅修改自Qwen2_5_VLForConditionalGeneration 的forward函数
        
        # 输入格式为 [batchSize * numPairs, seqLen]
        with torch.no_grad(): # visual encoder 不参与梯度更新
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids) # 0. 如果没嵌入的进行嵌入
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.dtype)
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # 1. 调用visual模块得到图像嵌入。
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                    mask = input_ids == self.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)

                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        
        # 默认batchsize不为None，也即成batch输入
        last_hidden_states = hidden_states[:, -1]  # [batch_size * num_pairs, hidden_size]
        scores = self.ranking_out_proj(self.ranking_dropout(self.activation_fn(self.ranking_dense(self.ranking_norm(last_hidden_states)))))  # [batch_size * num_pairs, 1]
        
        # 判断scores 一定可以整除self.config.pairNum
        assert scores.shape[0] % self.config.pairNum == 0, f"scores.shape[0] % self.config.pairNum != 0, scores.shape[0]: {scores.shape[0]}, self.config.pairNum: {self.config.pairNum}"
        
        scores = scores.view(-1, self.config.pairNum)  # 重塑为 [true_batch_size, num_pairs]

        # 1/50的概率输出scores，每个score*100保留两位小数
        if torch.rand(1).item() < 0.2:
            processed_scores = (scores * 100).round(decimals=2)
            print(f"处理后的分数: {processed_scores}")

        if labels is not None:
            # 主损失使用KL散度
            # loss1 = self.listnet_loss(scores, labels)
            # loss1 = MSELoss()(scores, labels)            
            
            cos_sim = F.cosine_similarity(scores, labels, dim=1).mean()
            loss1 = 1.0 - cos_sim 
            
            # 使用ranknet_loss作为辅助损失，确保正确的排序
            # ranknet损失现在直接使用原始分数，不需要sigmoid
            loss2 = self.config.lambdaOfLoss2 * self.ranknet_loss(scores, labels)
            
            # 总损失
            loss = loss1 + loss2
            
            # 计算准确率
            accuracy = self.calculate_accuracy(scores, labels)
        else:
            loss = None
            loss1 = None
            loss2 = None
            accuracy = None
        
        if not return_dict:
            output = (scores,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 返回selectorOutput，包含准确率
        return selectorOutput(
            loss=loss,
            loss1=loss1,
            loss2=loss2,
            logits=scores,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            accuracy=accuracy
        )

    def to_bf16(self):
        """将模型转换为bf16格式"""
        return self.to(torch.bfloat16)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 确保模型处于评估模式
        self.model.eval()
        # 现有代码...

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, default="./huggingface/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--numOfDecoderLayers", type=int, default=4)
    parser.add_argument("--lambdaOfLoss2", type=float, default=0.4)
    args = parser.parse_args()
    
    pretrained_model_name = args.pretrained_model_name
    print(f"Loading pretrained model and processor from: {pretrained_model_name}")

    processor = AutoProcessor.from_pretrained(pretrained_model_name, trust_remote_code=True)

    # 创建自定义配置
    config = selectorConfig.from_pretrained(
        pretrained_model_name, 
        trust_remote_code=True,
        pairNum=6,
        lambdaOfLoss2=args.lambdaOfLoss2,
        numOfDecoderLayers=args.numOfDecoderLayers
    )
    
    print("Custom config created with:")
    print(f"- pairNum: {config.pairNum}")
    print(f"- lambdaOfLoss2: {config.lambdaOfLoss2}")
    

    # 使用自定义配置加载模型
    model = selector.from_pretrained(
        pretrained_model_name,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16  # 通过dtype参数指定使用bf16
    )
    
    # 打印模型的结构
    print(model)
    
    # 检查一下 decoder 层的数量
    print(f"Number of decoder layers in the model: {len(model.model.layers)}")
    print("Model loaded and decoder layers pruned successfully.")

    # 打印模型的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params:,}")

    save_directory = f"./checkpoint/selectorInit-{args.numOfDecoderLayers}-{args.lambdaOfLoss2}"
    # 保存模型
    model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)
    config.save_pretrained(save_directory)
    print("Model, processor and config saved successfully in bf16 format.")
    
