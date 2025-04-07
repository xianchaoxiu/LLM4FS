import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import linear_model as lm
from sklearn import metrics as sm
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# -----------------------------
# 1. 读取特征重要性信息的函数
# -----------------------------
def load_feature_scores(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        gpt_data = json.load(f)

    feature_scores = []
    for obj in gpt_data:
        for key, value in obj.items():
            if key.startswith("concept"):
                feature_scores.append((value, obj["score"]))
                break

    # 按分数从高到低排序
    feature_scores_sorted = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    sorted_feature_names = [fs[0] for fs in feature_scores_sorted]
    return sorted_feature_names


# -----------------------------
# 2. 读取 one-hot 编码后的数据，并确定目标变量
# -----------------------------
df = pd.read_csv('./data/classical/Credit-G/Credit-G.csv')
y = df['Class']

# -----------------------------
# 3. 定义保存结果的字典
# -----------------------------
all_results = {
    "Feature Percentage": ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
    "DeepSeek-R1+RandomForest": []
}

# -----------------------------
# 4. 对不同特征比例进行实验（从10%到100%）
# -----------------------------
C_values = [0.1, 0.5, 1, 5, 10, 50, 100]
fractions = np.arange(0.1, 1.1, 0.1)

# 设置文件路径（根据需要切换）
file_paths = [

    ("DeepSeek-R1+RandomForest", "data/classical/Credit-G/deepseekR1+RandomForest_output.json")
]

# 对每个文件进行实验
for file_label, gpt_output_file in file_paths:
    sorted_feature_names = load_feature_scores(gpt_output_file)
    auc_values = []

    for frac in fractions:
        total_features = len(sorted_feature_names)
        n_features = math.ceil(frac * total_features)
        selected_concepts = sorted_feature_names[:n_features]

        # 针对每个概念，找到在 One-Hot 编码数据中所有以该概念为前缀的列
        selected_cols = []
        for concept_name in selected_concepts:
            matched_cols = [col for col in df.columns if col.startswith(concept_name)]
            selected_cols.extend(matched_cols)

        # 去重（若有概念前缀重叠的情况，避免重复列）
        selected_cols = list(set(selected_cols))

        # 如果一个概念都匹配不到列，则跳过
        if len(selected_cols) == 0:
            print(f"未能在数据中匹配到任何列，请检查概念名称：{selected_concepts}")
            continue

        # 从 one-hot 编码后的数据中选取对应的特征
        X_subset = df[selected_cols]

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)

        auroc_list = []

        # 重复实验 5 次（随机种子 1,2,3,4,5）
        for seed in [1, 2, 3, 4, 5]:
            train_x, test_x, train_y, test_y = ms.train_test_split(
                X_scaled, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
            )
            # 创建逻辑回归模型
            logistic = lm.LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100, random_state=seed)
            # 网格搜索（5折交叉验证）选取最佳超参数
            param_grid = {'C': C_values}
            grid_search = ms.GridSearchCV(logistic, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(train_x, train_y)
            best_logistic = grid_search.best_estimator_
            y_score = best_logistic.predict_proba(test_x)[:, 1]
            # 计算 ROC 曲线和 AUC
            fpr, tpr, _ = sm.roc_curve(test_y, y_score)
            roc_auc = sm.auc(fpr, tpr)
            auroc_list.append(roc_auc)

        avg_auc = np.mean(auroc_list)
        auc_values.append(avg_auc)

    # 保存当前文件的AUC值到结果字典
    all_results[file_label] = auc_values

# -----------------------------
# 5. 保存结果到 CSV 文件
# -----------------------------
results_df = pd.DataFrame(all_results)
results_df.set_index("Feature Percentage", inplace=True)

# 保存到CSV文件
results_df.to_csv("./data/classical/Credit-G/feature_selection_roc_results.csv")

# -----------------------------
# 6. 绘制ROC曲线图
# -----------------------------
plt.figure(figsize=(8, 6))
for file_label in all_results.keys():
    if file_label != "Feature Percentage":
        plt.plot(fractions * 100, all_results[file_label], label=file_label)

plt.xlabel("Percentage of Selected Features")
plt.ylabel("Average AUC (ROC)")
plt.title("Average AUC for Different Feature Subsets")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()