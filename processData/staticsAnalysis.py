original_data = [56, 8, 4, 0, 5, 551, 96, 155, 49, 76]
modified_data = [142, 38, 12, 0, 14, 534, 71, 103, 21, 65]

# 导入必要的库
from scipy.stats import spearmanr

# 计算Spearman相关系数
correlation, p_value = spearmanr(original_data, modified_data)

print(f"Spearman相关系数: {correlation:.4f}")
print(f"P值: {p_value:.4f}")

