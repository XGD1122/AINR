"""
共享配置文件：所有模型梯度实验的统一参数
"""
import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# DLPFC 数据路径 (相对于项目根目录)
DLPFC_DATA_DIR = os.path.join(PROJECT_ROOT, "DLPFC")
DLPFC_ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "DLPFC_annotations")
GASTON_DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# 结果保存路径
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============================================================
# DLPFC 切片配置
# ============================================================
# 所有 12 个 DLPFC 切片
ALL_SLICES = [
    '151507', '151508', '151509', '151510',
    '151669', '151670', '151671', '151672',
    '151673', '151674', '151675', '151676'
]

# 前4个切片 (用于 AINR 多切片训练，与原始实验一致)
AINR_TRAIN_SLICES = [151673, 151674, 151675, 151676]

# 每个切片的层数 (cluster 数量)
SLICE_N_CLUSTERS = {
    '151507': 7, '151508': 7, '151509': 7, '151510': 7,
    '151669': 5, '151670': 5, '151671': 5, '151672': 5,
    '151673': 7, '151674': 7, '151675': 7, '151676': 7,
}

def get_n_clusters(slice_id):
    """根据切片 ID 返回正确的层数"""
    sid = str(slice_id)
    return SLICE_N_CLUSTERS.get(sid, 7)

# ============================================================
# 实验公共参数
# ============================================================
DEFAULT_SEED = 112
DEFAULT_DEVICE = "cuda"  # 如果有 GPU 则用 cuda

# 梯度评估相关
GRADIENT_BATCH_SIZE = 2048

# 可视化基因列表 (与 Fig 3A 一致)
VIS_GENES = ['PCP4', 'CXCL14', 'ENC1', 'CCK', 'KRT17', 'MOBP']
