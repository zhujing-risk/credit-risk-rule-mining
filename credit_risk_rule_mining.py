# -*- coding: utf-8 -*-
"""
信贷风控规则挖掘完整流程
流程: 数据加载 → IV计算 → 特征筛选 → LightGBM建模 → 规则提取 → 规则评估
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import numpy as np
import pandas as pd
import os
import re
import time
import gc
import copy
import logging
import warnings
import multiprocessing
from datetime import datetime
from collections import Counter

# 机器学习
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

# 分箱与IV
from optbinning import OptimalBinning
from scipy.stats import spearmanr, kruskal, chi2_contingency

# 并行与进度
from joblib import Parallel, delayed
from tqdm import tqdm

# 绑定可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

# ODPS
import za_mlplatform_sdk as mlp
from odps.df import DataFrame

# 配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('optbinning').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# 中文字体
font_path = '/root/fonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第二部分：数据加载
# ============================================================
odps = mlp.get_odps_instance(data_id="data13a88922e07511f089a70242ac850002")

def read_data_from_odps(odpstablename):
    """从ODPS读取数据到pandas DataFrame"""
    table = DataFrame(odps.get_table(odpstablename))
    n_process = multiprocessing.cpu_count()
    print(f'n_process: {n_process}')
    dt1 = datetime.now()
    print(f'开始时间: {dt1.strftime("%Y-%m-%d %H:%M:%S")}')
    df = table.to_pandas(n_process=n_process)
    dt2 = datetime.now()
    print(f'结束时间: {dt2.strftime("%Y-%m-%d %H:%M:%S")}')
    duration = (dt2 - dt1).seconds
    print(f'耗时: {duration/60:.1f} 分钟 | shape: {df.shape}')
    return df


# ============================================================
# 第三部分：IV计算
# ============================================================
def detect_dtype(series):
    """自动检测列数据类型"""
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    return "categorical"


def preprocess_features(df, target_col):
    """预处理：过滤无效列"""
    valid_cols = []
    dtypes = {}

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isnull().mean() > 0.8:
            continue
        if df[col].nunique() <= 1:
            continue
        valid_cols.append(col)
        dtypes[col] = detect_dtype(df[col])

    return valid_cols, dtypes


def calculate_iv_single(col, df_subset):
    """快速计算单个变量的IV"""
    try:
        if len(df_subset) < 50:
            return None

        feature_col = df_subset.columns[0]
        target_col = df_subset.columns[1]
        feature_series = df_subset[feature_col]
        target_series = df_subset[target_col]

        if pd.api.types.is_numeric_dtype(feature_series):
            n_bins = min(10, len(df_subset) // 50)
            try:
                bins = pd.qcut(feature_series, q=n_bins, duplicates='drop', labels=False)
            except:
                bins = pd.cut(feature_series, bins=n_bins, labels=False, duplicates='drop')
            dtype = 'numeric'
        else:
            bins = feature_series.astype('category').cat.codes
            dtype = 'categorical'

        # IV计算
        df_temp = pd.DataFrame({'target': target_series.values, 'bins': bins.values}).dropna()
        if len(df_temp) < 2:
            return None

        grouped = df_temp.groupby('bins')['target'].agg(['count', 'sum'])
        grouped['non_events'] = grouped['count'] - grouped['sum']

        total_events = grouped['sum'].sum()
        total_non_events = grouped['non_events'].sum()
        if total_events == 0 or total_non_events == 0:
            return None

        # 拉普拉斯平滑
        grouped['event_pct'] = (grouped['sum'] + 0.5) / (total_events + 1)
        grouped['non_event_pct'] = (grouped['non_events'] + 0.5) / (total_non_events + 1)
        grouped['woe'] = np.log(grouped['event_pct'] / grouped['non_event_pct'])
        iv = ((grouped['event_pct'] - grouped['non_event_pct']) * grouped['woe']).sum()

        return {'variable': col, 'iv': iv, 'dtype': dtype}
    except Exception as e:
        return None


def compute_iv_fast(df, target_col='dob4_ever10_flg'):
    """并行计算所有特征的IV"""
    cols = [col for col in df.columns if col != target_col]
    missing_rates = df[cols].isnull().mean()
    nunique_vals = df[cols].nunique()

    valid_cols = [col for col in cols
                  if missing_rates[col] <= 0.8 and nunique_vals[col] > 1]

    results = Parallel(n_jobs=-1)(
        delayed(calculate_iv_single)(col, df[[col, target_col]].dropna())
        for col in tqdm(valid_cols, desc="IV计算")
    )

    iv_df = pd.DataFrame([r for r in results if r is not None])
    if not iv_df.empty:
        iv_df = iv_df.sort_values('iv', ascending=False).reset_index(drop=True)
        iv_df['strength'] = np.select(
            [iv_df['iv'] <= 0.02, iv_df['iv'] <= 0.1, iv_df['iv'] <= 0.3, iv_df['iv'] > 0.3],
            ['无预测力', '弱', '中等', '强'],
            default='无预测力'
        )
    return iv_df


# ============================================================
# 第四部分：变量分析（输出PDF）
# ============================================================
def var_analysis(raw_data, variables, target_variable, var_month, out_pdf):
    """
    按月分组绘制变量的分箱分布和坏客户占比
    raw_data: 宽表DataFrame
    variables: 要分析的变量列表
    target_variable: 目标变量
    var_month: 月份变量
    out_pdf: 输出PDF文件名
    """
    months = sorted(raw_data[var_month].unique())
    nmonth = len(months) + 1

    with PdfPages(out_pdf) as pdf:
        for var in tqdm(variables, desc="绘制变量分析"):
            try:
                fig, axes = plt.subplots(1, nmonth, figsize=(20, 5))

                for i, month in enumerate(months):
                    monthly_data = raw_data[raw_data[var_month] == month].copy()
                    x = monthly_data[var]
                    monthly_data['bin'] = pd.qcut(x, q=10, duplicates='drop')

                    grouped = monthly_data.groupby('bin')[target_variable].agg(['count', 'sum'])
                    grouped.columns = ['total', 'bad']
                    grouped['good'] = grouped['total'] - grouped['bad']
                    grouped['bad_rate'] = grouped['bad'] / grouped['total']

                    total_good = grouped['good'].sum()
                    total_bad = grouped['bad'].sum()
                    grouped['good_r'] = grouped['good'] / total_good
                    grouped['bad_r'] = grouped['bad'] / total_bad
                    grouped['woe'] = np.log(grouped['good_r'] / grouped['bad_r'])
                    grouped['iv'] = (grouped['good_r'] - grouped['bad_r']) * grouped['woe']
                    iv_value = grouped['iv'].sum()

                    ax1 = axes[i]
                    ax1.bar(grouped.index.astype(str), grouped['total'], color='skyblue')
                    ax1.set_xlabel('分组')
                    ax1.set_ylabel('客户频数', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    ax1.set_title(f'{var} - {month} (IV={iv_value:.4f})', fontsize=12, fontweight='bold')
                    ax1.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')

                    ax2 = ax1.twinx()
                    ax2.plot(grouped.index.astype(str), grouped['bad_rate'], color='red', marker='o')
                    ax2.set_ylabel('坏客户占比', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    for j, bad_rate in enumerate(grouped['bad_rate']):
                        ax2.text(j, bad_rate + 0.002, f'{bad_rate*100:.2f}%', ha='center', va='bottom', fontsize=9)

                # 总图
                x = raw_data[var]
                raw_data_tmp = raw_data.copy()
                raw_data_tmp['bin'] = pd.qcut(x, q=10, duplicates='drop')
                grouped = raw_data_tmp.groupby('bin')[target_variable].agg(['count', 'sum'])
                grouped.columns = ['total', 'bad']
                grouped['good'] = grouped['total'] - grouped['bad']
                grouped['bad_rate'] = grouped['bad'] / grouped['total']
                total_good = grouped['good'].sum()
                total_bad = grouped['bad'].sum()
                grouped['good_r'] = grouped['good'] / total_good
                grouped['bad_r'] = grouped['bad'] / total_bad
                grouped['woe'] = np.log(grouped['good_r'] / grouped['bad_r'])
                grouped['iv'] = (grouped['good_r'] - grouped['bad_r']) * grouped['woe']
                iv_value = grouped['iv'].sum()

                ax1 = axes[-1]
                ax1.bar(grouped.index.astype(str), grouped['total'], color='skyblue')
                ax1.set_xlabel('分组')
                ax1.set_ylabel('客户频数', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.set_title(f'{var} - 总图 (IV={iv_value:.4f})', fontsize=12, fontweight='bold')
                ax1.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')

                ax2 = ax1.twinx()
                ax2.plot(grouped.index.astype(str), grouped['bad_rate'], color='red', marker='o')
                ax2.set_ylabel('坏客户占比', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                for j, bad_rate in enumerate(grouped['bad_rate']):
                    ax2.text(j, bad_rate + 0.002, f'{bad_rate*100:.2f}%', ha='center', va='bottom', fontsize=9)

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close()
            except Exception as e:
                print(f"Error processing variable {var}: {e}")


# ============================================================
# 第五部分：单调性分析
# ============================================================
def cochran_armitage_test(bad_rates, bin_counts):
    """Cochran-Armitage趋势检验"""
    try:
        contingency_table = np.array([
            bin_counts * (1 - bad_rates),
            bin_counts * bad_rates
        ]).T.astype(int)

        if np.any(contingency_table < 5):
            return np.nan

        n_rows = contingency_table.shape[0]
        weights = np.arange(1, n_rows + 1)
        weighted_table = contingency_table * weights[:, None]
        chi2, pval, dof, expected = chi2_contingency(weighted_table)
        return pval
    except:
        return np.nan


def detect_trend_direction(bad_rates):
    """自动判断趋势方向"""
    if len(bad_rates) < 2:
        return 'none'
    differences = bad_rates.diff().dropna()
    sign_sum = (differences > 0).sum() - (differences < 0).sum()
    if sign_sum > 0:
        return 'increase'
    elif sign_sum < 0:
        return 'decrease'
    return 'stable'


def analyze_variable(df, x_col, y_col, n_bins=10, trend='auto', min_bin_size=0.02):
    """分析单个变量的单调性"""
    try:
        df = df.copy()
        df['bin'], bins = pd.qcut(df[x_col], q=n_bins, duplicates='drop', retbins=True)
        bad_rates = df.groupby('bin')[y_col].mean().sort_index()
        bin_counts = df.groupby('bin').size()

        valid_mask = (bin_counts / len(df)) >= min_bin_size
        bad_rates = bad_rates[valid_mask]
        bin_counts = bin_counts[valid_mask]
        if len(bad_rates) < 2:
            return None

        actual_trend = detect_trend_direction(bad_rates) if trend == 'auto' else trend
        if actual_trend == 'none':
            return None

        bin_ranks = np.arange(len(bad_rates))
        rho, _ = spearmanr(bin_ranks, bad_rates)
        violations = (sum(np.diff(bad_rates) < 0) if actual_trend == 'increase'
                      else sum(np.diff(bad_rates) > 0))

        return {
            'variable': x_col,
            'n_bins': len(bad_rates),
            'bad_rate_min': bad_rates.min(),
            'bad_rate_max': bad_rates.max(),
            'trend_direction': actual_trend,
            'trend_strength': abs(rho),
            'violation_ratio': violations / (len(bad_rates) - 1),
            'cochran_armitage_p': cochran_armitage_test(bad_rates.values, bin_counts.values),
            'bins': bins.tolist()
        }
    except Exception as e:
        return None


def batch_analyze(df, x_cols, y_col, nbins=10, trend='auto', min_bin_size=0.02, n_jobs=4):
    """并行批量单调性分析"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze_variable)(df, col, y_col, n_bins=nbins, trend=trend, min_bin_size=min_bin_size)
        for col in tqdm(x_cols, desc="单调性分析")
    )
    return pd.DataFrame([r for r in results if r is not None])


# ============================================================
# 第六部分：模型评估工具函数
# ============================================================
def plot_matrix_report(y_label, y_pred):
    """混淆矩阵"""
    matrix_array = metrics.confusion_matrix(y_label, y_pred)
    plt.matshow(matrix_array, cmap=plt.cm.summer_r)
    plt.colorbar()
    for x in range(len(matrix_array)):
        for y in range(len(matrix_array)):
            plt.annotate(matrix_array[x, y], xy=(x, y), ha='center', va='center')
    plt.xlabel('True label')
    plt.ylabel('Predict label')
    print(metrics.classification_report(y_label, y_pred))
    plt.show()


def PlotKS(preds, labels, n=10000, asc=0):
    """KS曲线"""
    pred = preds
    bad = labels
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    else:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    else:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print(f'ks_value is {ks_value:.4f} at pop = {ks_pop:.4f}')

    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good', color='blue', linestyle='-', linewidth=2)
    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad', color='red', linestyle='-', linewidth=2)
    plt.plot(ksds.tile, ksds.ks, label='ks', color='green', linestyle='-', linewidth=2)
    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.title(f'KS={ks_value:.4f} at Pop={ks_pop:.4f}', fontsize=15)
    plt.legend()
    plt.show()


def PlotROC(preds, labels):
    """ROC曲线"""
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC={auc_score:.5f}')
    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax.legend()
    plt.show()


def m_plot(model, X, y, threshold=0.5, n=1000, asc=0):
    """模型综合评估：混淆矩阵 + KS + ROC"""
    y_predicted = model.predict(X)
    y_pred = [int(v > threshold) for v in y_predicted]
    plot_matrix_report(y, y_pred)
    PlotKS(y_predicted, y, n=n, asc=asc)
    PlotROC(y_predicted, y)


# ============================================================
# 第七部分：LightGBM训练
# ============================================================
def train_lgb_model(train_data_x, train_data_y):
    """两阶段LightGBM训练"""
    X_train, X_val, y_train, y_val = train_test_split(
        train_data_x, train_data_y,
        test_size=0.3, random_state=42, stratify=train_data_y
    )

    train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, free_raw_data=False)

    base_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['auc', 'binary_error'],
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 1000,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'verbose': -1
    }

    # 第一阶段：GBDT快速收敛
    print("=== 第一阶段训练 (GBDT) ===")
    stage1_params = base_params.copy()
    stage1_params['learning_rate'] = 0.1

    model = lgb.train(
        stage1_params,
        train_set,
        num_boost_round=200,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'valid'],
        callbacks=[lgb.log_evaluation(10), lgb.early_stopping(50)]
    )

    # 第二阶段：DART精调
    print("\n=== 第二阶段训练 (DART) ===")
    stage2_params = base_params.copy()
    stage2_params.update({
        'learning_rate': 0.02,
        'boosting_type': 'dart'
    })

    model = lgb.train(
        stage2_params,
        train_set,
        num_boost_round=300,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'valid'],
        callbacks=[lgb.log_evaluation(10), lgb.early_stopping(50)]
    )

    # 评估
    val_pred = model.predict(X_val)
    print(f"\n验证集 AUC: {roc_auc_score(y_val, val_pred):.4f}")
    m_plot(model, X_val, y_val, threshold=0.5)

    return model, X_train, X_val, y_train, y_val


# ============================================================
# 第八部分：规则提取（优化版）
# ============================================================
def extract_decision_paths_fast(model, feature_names, sample_data, max_depth=4):
    """从LightGBM模型树中提取决策路径规则"""
    leaf_ids = model.predict(sample_data, pred_leaf=True)
    tree_dicts = model.dump_model()['tree_info']

    # 预构建所有树的路径映射
    tree_paths = []
    for tree_info in tree_dicts:
        node_paths = {}
        stack = [(tree_info['tree_structure'], [], 0)]
        while stack:
            node, path, depth = stack.pop()
            if depth >= max_depth or 'split_feature' not in node:
                if 'leaf_index' in node:
                    node_paths[node['leaf_index']] = ' AND '.join(path)
                continue

            feat_name = feature_names[node['split_feature']]
            thresh = node['threshold']
            fmt = f"{thresh:.4f}" if isinstance(thresh, float) else str(thresh)

            if 'left_child' in node:
                stack.append((node['left_child'], path + [f"{feat_name}<={fmt}"], depth + 1))
            if 'right_child' in node:
                stack.append((node['right_child'], path + [f"{feat_name}>{fmt}"], depth + 1))

        tree_paths.append(node_paths)

    # 批量映射leaf_id到规则
    rules = []
    for tree_idx in range(leaf_ids.shape[1]):
        path_map = tree_paths[tree_idx]
        for leaf_id in leaf_ids[:, tree_idx]:
            r = path_map.get(leaf_id)
            if r:
                rules.append(r)

    return pd.Series(rules).value_counts()


# ============================================================
# 第九部分：规则评估（优化版）
# ============================================================
# 预编译正则：匹配 "变量名 操作符 值"
_CONDITION_RE = re.compile(r'^(.+?)\s*(<=|>=|!=|==|>|<)\s*(.+)$')


def parse_conditions(rule_str):
    """解析规则字符串为 [(var, op, val), ...] 列表"""
    normalized = rule_str.replace('≤', '<=').replace('≥', '>=').strip()
    result = []
    for cond in normalized.split(' AND '):
        m = _CONDITION_RE.match(cond.strip())
        if m:
            result.append((m.group(1).strip(), m.group(2), m.group(3).strip()))
    return result


def should_exclude_rule(rule_str, good_vars_set, bad_vars_set):
    """
    排除方向性不合理的规则：
    - good_vars（越大越好）出现 > 或 >= → 选的是好人，排除
    - bad_vars（越大越坏）出现 < 或 <= → 选的是好人，排除
    """
    for var, op, _ in parse_conditions(rule_str):
        if var in good_vars_set and op in ('>', '>='):
            return True
        if var in bad_vars_set and op in ('<', '<='):
            return True
    return False


def _eval_single_rule(rule, df_values, col_index, label_arr,
                      n_total, total_bad_rate, min_cov, max_cov, min_bads):
    """单条规则的向量化评估（纯numpy）"""
    conditions = parse_conditions(rule)
    if not conditions:
        return None

    mask = np.ones(n_total, dtype=bool)

    for var, op, val_str in conditions:
        idx = col_index.get(var)
        if idx is None:
            return None
        col = df_values[:, idx]
        try:
            val = float(val_str)
        except ValueError:
            return None

        if op == '<=':
            mask &= col <= val
        elif op == '>=':
            mask &= col >= val
        elif op == '>':
            mask &= col > val
        elif op == '<':
            mask &= col < val
        elif op == '==':
            mask &= col == val
        elif op == '!=':
            mask &= col != val

        # 提前剪枝
        if not mask.any():
            return None

    hit = mask.sum()
    coverage = hit / n_total

    if coverage < min_cov or coverage > max_cov:
        return None

    bads = label_arr[mask].sum()
    if bads < min_bads:
        return None

    bad_rate = bads / hit

    return {
        '原始规则': rule,
        '命中样本': int(hit),
        '坏样本数': int(bads),
        '覆盖率': f"{coverage:.2%}",
        '坏账率': f"{bad_rate:.2%}",
        '提升度': f"{bad_rate / total_bad_rate:.2f}" if total_bad_rate > 0 else "0",
        'bad_rate': bad_rate
    }


def auto_detect_good_vars(df_columns, user_good_vars=None):
    """自动识别good_vars：含model+score的特征"""
    auto = [c for c in df_columns
            if 'model' in c.lower() and 'score' in c.lower()]
    return list(set((user_good_vars or []) + auto))


def auto_detect_bad_vars(df_columns, user_bad_vars=None, exclude_vars=None):
    """自动识别bad_vars：含num/cnt/org/amt的特征"""
    exclude = set(exclude_vars or [])
    pattern = re.compile(r'(num|cnt|org|amt)', re.IGNORECASE)
    auto = [c for c in df_columns if pattern.search(c) and c not in exclude]
    return list(set((user_bad_vars or []) + auto))


def evaluate_rules(
    df,
    rules,
    label_col='dob4_ever10_flg',
    good_vars=None,
    bad_vars=None,
    min_coverage=0.05,
    max_coverage=0.50,
    min_bads=50,
    n_jobs=-1
):
    """
    优化版规则评估主函数
    1. 自动识别good_vars/bad_vars
    2. 预过滤方向性不合理规则
    3. numpy向量化 + 并行评估
    """
    # 构建good/bad变量集合
    all_good_vars = auto_detect_good_vars(df.columns, good_vars)
    all_bad_vars = auto_detect_bad_vars(df.columns, bad_vars, exclude_vars=all_good_vars)
    good_vars_set = set(all_good_vars)
    bad_vars_set = set(all_bad_vars)

    print(f"good_vars: {len(good_vars_set)} 个 | bad_vars: {len(bad_vars_set)} 个")

    # 预过滤
    valid_rules = [r for r in rules if not should_exclude_rule(r, good_vars_set, bad_vars_set)]
    print(f"规则过滤: {len(rules)} → {len(valid_rules)} (排除 {len(rules)-len(valid_rules)} 条)")

    # 收集规则中涉及的列
    all_vars_in_rules = set()
    for r in valid_rules:
        for var, _, _ in parse_conditions(r):
            all_vars_in_rules.add(var)

    used_cols = [c for c in df.columns if c in all_vars_in_rules]
    col_index = {c: i for i, c in enumerate(used_cols)}

    # 转numpy（只转规则涉及的列）
    df_numeric = df[used_cols].copy()
    for col in used_cols:
        if not pd.api.types.is_numeric_dtype(df_numeric[col]):
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    df_values = df_numeric.values.astype(np.float64)
    label_arr = df[label_col].values.astype(np.float64)
    n_total = len(df)
    total_bad_rate = label_arr.mean()

    print(f"基准坏账率: {total_bad_rate:.2%} | 样本量: {n_total}")
    print(f"覆盖率范围: {min_coverage:.1%} - {max_coverage:.1%}")

    # 并行评估
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_eval_single_rule)(
            rule, df_values, col_index, label_arr,
            n_total, total_bad_rate, min_coverage, max_coverage, min_bads
        )
        for rule in tqdm(valid_rules, desc="评估规则")
    )

    results = [r for r in results if r is not None]

    if not results:
        print("未找到满足条件的规则")
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values('bad_rate', ascending=False).reset_index(drop=True)
    result_df['bad_rate'] = result_df['bad_rate'].apply(lambda x: f"{x:.4f}")

    print(f"最终有效规则: {len(result_df)} 条")
    return result_df


# ============================================================
# 第十部分：执行主流程
# ============================================================
if __name__ == '__main__':

    # ---------- 1. 加载数据 ----------
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    df = read_data_from_odps('yy_apply_kb_zj_05_nobr_xf_pass_rule_01_202506_0617')
    print(f"数据shape: {df.shape}")
    print(df.head(3))

    # ---------- 2. 定义排除字段 ----------
    to_drop = [
        'is_perform_fpb1', 'is_perform_fpb5', 'is_perform_fpb10', 'is_perform_fpb31',
        'is_perform_dob2_ever10', 'is_perform_dob3_ever10', 'is_perform_dob4_ever10',
        'is_perform_dob5_ever10', 'is_perform_dob6_ever10',
        'fpd1_flag', 'fpd5_flag', 'fpd10_flag', 'fpd31_flag',
        'dob2_ever10_flg', 'dob3_ever10_flg', 'dob4_ever10_flg', 'dob5_ever10_flg', 'dob6_ever10_flg',
        'dob2_ever30_flg', 'dob3_ever30_flg', 'dob4_ever30_flg', 'dob5_ever30_flg', 'dob6_ever30_flg',
        'fpd1_amt', 'fpd5_amt', 'fpd10_amt', 'fpd31_amt',
        'dob2_ever10_amt', 'dob3_ever10_amt', 'dob4_ever10_amt', 'dob5_ever10_amt', 'dob6_ever10_amt',
        'dob2_ever30_amt', 'dob3_ever30_amt', 'dob4_ever30_amt', 'dob5_ever30_amt', 'dob6_ever30_amt',
        'channel_type_final', 'product_code', 'loan_apply_no', 'apply_month',
        'credit_apply_date', 'credit_amount',
        'rule_new', 'hit_rule_new', 'hit_rule_new2',
        'now_credit_amount', 'trans_withdraw_amount',
        'dxm_general_prea_consc_v2_score', 'user_id',
        'pd_ylzc_shouyufen_upsd001_score', 'pd_zj_25088_score',
        'td_i_cnt_node_dist2_loan_all_all',
        'tcxy_creditpro_v1_repaysuc_cnt_360d_ratio_by_trans_cnt_360d',
        'pudao_jig_credit_cs1_score', 'pd_txty_hrate7_score',
        'pd_baiduyun_duyifen_a_02_8', 'pd_jd_xuanyuan_plus',
        'pd_gd_za_cust_score_v1', 'pd_blz_tl_yh_v1', 'pd_dxm_general_prea_v10',
        'txty_total_xe4_5_score', 'jbx_scoree_score_e',
        'pd_ylzc_shouyufen_upsd001_score', 'pd_ylzc_shouyufen_upsd002_score',
        'credit_risk_score_x2_bys', 'credit_risk_score_x3_bys',
        'tcxy_applypro_v1_apply_model_score_high',
        # model_jdzad系列
        'model_jdzad_ziac2v026_a0t130m0_score', 'model_jdzad_ziac2v025_a0t110m0_score',
        'model_jdzad_zaac2v023_a0t130m0_score', 'model_jdzad_zaac2v022_a0t110m0_score',
        'model_jdzad_zaac1v44_g0t110x0_score', 'model_jdzad_ziac1v022_p0m210x_score',
        'model_jdzad_zaec1v42_00m430m1_score', 'model_jdzad_zasc1v41_00m430m1_score',
        'model_jdzad_ziac1v019_p0m230x0_score', 'model_jdzad_zaac1v37_a0m430xm_score',
        'model_jdzad_zaac1v34_a0t110x0_score', 'model_jdzad_zaac1v33_a0m330x0_score',
        'model_jdzad_zaac1v32_f0m430l0_score', 'model_jdzad_ziac1v020_w0nt05x0_score',
        'model_jdzad_zaac1v30_a0m430x0_score', 'model_jdzad_zaac1v31_a0t110x0_score',
        'model_jdzad_zaac1v29_a0m430x0_score', 'model_jdzad_zaac1v28_a0m430x0_score',
        'model_jdzad_zafc1v028_a0m430x0_score', 'model_jdzad_zafc1v027_a0t110x0_score',
        'model_jdzad_zaac1v26_a0m430x0_score', 'model_jazad_zaac1v25_f0m430x0_score',
        'model_jdzad_zisc1v019_p0t15l0_score', 'model_jdzad_zaac1v020_f0m310x0_score',
        'model_jdzad_ziac1v013_p0nt30x0_score', 'model_jdzad_zifc1v015_z0nt05xm_score',
        'model_jdzad_ziac1v012_p0nt30x0_score', 'model_jdzad_ziac1v011_p0t110x0_score',
        'model_jdzad_ziac1v009_z0nt03x0_score', 'model_jdzad_acard_ylzc_score',
        'model_jdzad_ziac1v019_p0m230x0_score2', 'model_jdzad_ziac1v020_p0m230x0_score2',
        'model_jdzad_zaec1v42_00m430m0_score', 'model_jdzad_zasc1v41_00m430m0_score',
        'model_jdzad_zaac1v49_c0m430x0_score', 'model_jdzad_zafc2v024_g0t110x0_score',
        'model_jdzad_zaac1v39_a0m430x0_score', 'model_jdzad_zifc1v017_z0nt05x0_score',
        'model_jdzad_zifc1v016_z0nt05x0_score', 'model_jdzad_acard_zaac00v97_9xlx_score',
        'pd_zj_17178_score',
    ]

    # ---------- 3. 计算IV ----------
    print("\n" + "=" * 60)
    print("Step 2: 计算IV")
    print("=" * 60)
    raw_data = df[df['is_perform_dob4_ever10'] > 0].copy()
    iv_result = compute_iv_fast(raw_data.drop(columns=[c for c in to_drop if c in raw_data.columns], errors='ignore'),
                                'dob4_ever10_flg')
    print(iv_result.head(20))

    # ---------- 4. 准备建模数据 ----------
    print("\n" + "=" * 60)
    print("Step 3: 准备建模数据")
    print("=" * 60)
    dftmp = df[df['is_perform_dob4_ever10'] > 0].copy()

    scoreVar = [x for x in dftmp.columns
                if x.find('td_') > -1 or x.find('score') > -1 or x.find('bh_') > -1]

    train_data_x = dftmp[list(set(scoreVar) - set(to_drop))].copy()
    train_data_y = dftmp['dob4_ever10_flg']

    # 类型转换
    var_char = train_data_x.dtypes[train_data_x.dtypes == 'object'].index.tolist()
    for col in var_char:
        try:
            train_data_x[col] = pd.to_numeric(train_data_x[col])
        except ValueError:
            train_data_x[col] = train_data_x[col].astype('category')

    print(f"建模特征数: {train_data_x.shape[1]} | 样本量: {train_data_x.shape[0]}")

    # ---------- 5. 训练模型 ----------
    print("\n" + "=" * 60)
    print("Step 4: LightGBM训练")
    print("=" * 60)
    lgbmodel, X_train, X_val, y_train, y_val = train_lgb_model(train_data_x, train_data_y)

    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': lgbmodel.feature_name(),
        'importance': lgbmodel.feature_importance()
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    print("\nTop 10 特征重要性:")
    print(importance_df.head(10))

    # ---------- 6. 提取规则 ----------
    print("\n" + "=" * 60)
    print("Step 5: 规则提取")
    print("=" * 60)
    top_rules = extract_decision_paths_fast(
        lgbmodel,
        feature_names=train_data_x.columns.tolist(),
        sample_data=train_data_x,
        max_depth=4
    )
    print(f"提取规则总数: {len(top_rules)}")

    # ---------- 7. 规则评估 ----------
    print("\n" + "=" * 60)
    print("Step 6: 规则评估")
    print("=" * 60)

    # 定义good_vars（越大越好的特征）
    good_vars = [
        "bh_br_scoreysstd", "bh_xys_byf_linglong_score82",
        "dxm_general_prea_consc_v2_score", "jbx_lhjm_v1_acard_score",
        "jbx_scorec_score_c", "pd_blz_tl_yh_v1", "pd_bwtj_score_v5_1",
        "pd_dxm_general_prea_v10", "pd_dxm_xmscore_xiaodaiv11_am6",
        "pd_hn_bczv3_score", "pd_jd_zacxdz3_has_child_score",
        "pd_rong360_acard_dz_score_v2", "pd_shhj_chfs8_3_score",
        "pd_shhj_chfs8_6_score", "pd_tc_puchen_score", "pd_td_td111_score",
        "pd_xyd_bee_score_rate_17_score", "pd_yr_lm_score",
        "td_large_cash_score", "xys_total_qarnet_score46",
        "xysl_lanyu_score1", "yd_al_xiaoniu_scorea1", "zzx_pboc2_score",
        "tcxy_creditpro_v1_credit_model_score_high",
        "pd_rong360_acard_dz_score", "pboc2_pb_dtl_r1_bdi_acctcl_sum_rt",
        "pd_dxm_dongzhi_score_v1",
        "tcxy_applypro_v1_mobile_not_verification_cur_diffdays_max",
        "pd_mayi_aft_v3_score",
    ]

    # 定义bad_vars（越大越坏的特征）
    bad_vars = [
        "baidu_panshi_multiloans_score", "bh_aly_anti_fraud_v6_score",
        "bh_hp_credit_operat_score", "bh_hp_custom_credit_score_y",
        "bh_hp_customer_seg_score", "bh_unionpay_hy_score",
        "bwjk_relationship_network_score", "pd_baiduyun_duyifen_a_02_8",
        "pd_bdy_finance_fraud_score", "pd_blz_ttc_score",
        "pd_dxm_zhonganbxa_v2", "pd_haluo_insightv7_score",
        "pd_jg_score_credit_s3v1", "pd_mayi_aft_v3_score",
        "pd_mob_v129_score", "pd_my_hl_aft_v4_score",
        "pd_shhj_chfs8_4_score", "pd_txty_hrate7_score",
        "rong360_fraud_riskscore", "td_fraud_score",
        "txty_total_de4_score", "txty_total_xe4_5_score",
        "pd_rong360_acard_dz_score_p", "pboc2_pb_dtl_crdt_cfc_usdcl_sum",
    ]

    train_df = pd.concat([train_data_x, train_data_y], axis=1)

    rule_report = evaluate_rules(
        df=train_df,
        rules=top_rules.index,
        label_col='dob4_ever10_flg',
        good_vars=good_vars,
        bad_vars=bad_vars,
        min_coverage=0.10,
        max_coverage=0.50,
        min_bads=50,
        n_jobs=-1
    )

    print("\n" + "=" * 60)
    print("最终规则报告 (Top 20):")
    print("=" * 60)
    print(rule_report.head(20).to_markdown(index=False))
