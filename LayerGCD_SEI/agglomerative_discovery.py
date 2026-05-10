import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import gaussian_filter1d


def estimate_optimal_threshold(feats, n_sample_pairs=50000, bins=200, smooth_sigma=3.0):
    """
    自动从特征分布中估算最优合并距离阈值（无监督，不需要任何标签）。

    原理：
      随机采样大量特征对，计算余弦距离分布。对于训练充分的网络而言，
      同类样本对的距离集中在左侧（小距离），不同类样本对集中在右侧（大距离），
      中间会形成一个波谷（Valley）。找到这个波谷的位置，即为最优 threshold。

    Args:
        feats (np.ndarray): L2-normalized feature embeddings, shape (N, D).
        n_sample_pairs (int): 随机采样的对数（避免 O(N²) 计算）。
        bins (int): 直方图分箱数。
        smooth_sigma (float): 对分布做高斯平滑的 sigma，用于抑制噪声再找极值。

    Returns:
        threshold (float): 推荐的 merge_threshold（余弦距离）。
        hist_info (dict): 包含分布数据，用于可视化调试。
    """
    N = len(feats)
    
    # 随机采样对（避免 O(N²) 全量计算）
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, N, size=n_sample_pairs)
    idx_b = rng.integers(0, N, size=n_sample_pairs)
    same_pair = idx_a == idx_b
    idx_b[same_pair] = (idx_b[same_pair] + 1) % N
    
    # 计算余弦距离 = 1 - cosine_similarity
    fa = feats[idx_a]  # (P, D)
    fb = feats[idx_b]  # (P, D)
    cos_sim = np.sum(fa * fb, axis=1)  # 因为已经 L2 归一化
    cos_dist = 1.0 - cos_sim
    cos_dist = np.clip(cos_dist, 0.0, 2.0)
    
    # 构建直方图
    hist, bin_edges = np.histogram(cos_dist, bins=bins, range=(0.0, 2.0), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 高斯平滑（去掉采样噪声）
    hist_smooth = gaussian_filter1d(hist, sigma=smooth_sigma)
    
    # 寻找"第一个局部最小值"：
    # 跳过从 0 开始的上升沿，在第一个峰（intra-class 峰）之后找第一个谷底
    # 策略：先找第一个局部极大值，再往右找最近的局部极小值
    first_peak_idx = np.argmax(hist_smooth)  # 全局最大值（通常是 intra 峰）
    
    # 从 first_peak_idx 开始向右搜索第一个谷底
    valley_idx = first_peak_idx
    for i in range(first_peak_idx + 1, len(hist_smooth) - 1):
        if hist_smooth[i] < hist_smooth[i - 1] and hist_smooth[i] < hist_smooth[i + 1]:
            valley_idx = i
            break
            
    # Average linkage uses cluster-level mean distances, so a small margin around
    # the pairwise valley keeps the threshold robust to seed-level fluctuation.
    threshold = float(bin_centers[valley_idx]) + 0.10
    
    # 安全边界：不能太小（< 0.05 说明网络完全未训练）也不能太大
    threshold = np.clip(threshold, 0.05, 1.5)
    
    hist_info = {
        'bin_centers': bin_centers,
        'hist': hist,
        'hist_smooth': hist_smooth,
        'first_peak_idx': first_peak_idx,
        'valley_idx': valley_idx,
        'n_sample_pairs': n_sample_pairs,
    }
    
    return threshold, hist_info


def estimate_known_calibrated_threshold(
    feats,
    labels,
    n_sample_pairs=50000,
    intra_quantile=0.95,
    inter_quantile=0.05,
):
    """
    Estimate a cosine-distance merge threshold from labeled known emitters only.

    The threshold does not use novel-emitter labels or the total number of
    classes. It calibrates the maximum within-emitter spread that HAC should
    preserve before applying discovery to mixed known/novel data.
    """
    feats = np.asarray(feats)
    labels = np.asarray(labels)
    if len(feats) < 2 or len(np.unique(labels)) < 2:
        return 0.25, {
            "intra_q": 0.25,
            "inter_q": 0.25,
            "n_intra": 0,
            "n_inter": 0,
        }

    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, len(feats), size=n_sample_pairs)
    idx_b = rng.integers(0, len(feats), size=n_sample_pairs)
    same_index = idx_a == idx_b
    idx_b[same_index] = (idx_b[same_index] + 1) % len(feats)

    fa = feats[idx_a]
    fb = feats[idx_b]
    cos_dist = np.clip(1.0 - np.sum(fa * fb, axis=1), 0.0, 2.0)
    same_label = labels[idx_a] == labels[idx_b]
    intra = cos_dist[same_label]
    inter = cos_dist[~same_label]
    if len(intra) == 0 or len(inter) == 0:
        return 0.25, {
            "intra_q": 0.25,
            "inter_q": 0.25,
            "n_intra": int(len(intra)),
            "n_inter": int(len(inter)),
        }

    intra_q = float(np.quantile(intra, intra_quantile))
    inter_q = float(np.quantile(inter, inter_quantile))
    if intra_q < inter_q:
        threshold = 0.75 * intra_q
    else:
        threshold = 0.90 * inter_q
    threshold = float(np.clip(threshold, 0.02, 1.50))
    return threshold, {
        "intra_q": intra_q,
        "inter_q": inter_q,
        "n_intra": int(len(intra)),
        "n_inter": int(len(inter)),
    }


def merge_small_clusters(feats, labels, min_cluster_size=None):
    if len(labels) == 0:
        return labels
    if min_cluster_size is None:
        min_cluster_size = max(20, int(0.03 * len(labels)))

    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= min_cluster_size]
    small = unique[counts < min_cluster_size]
    if len(large) == 0 or len(small) == 0:
        return labels

    centers = {}
    for clus_id in large:
        idx = labels == clus_id
        center = feats[idx].mean(axis=0)
        norm = np.linalg.norm(center)
        centers[clus_id] = center / norm if norm > 1e-8 else center

    merged = labels.copy()
    for clus_id in small:
        idx = labels == clus_id
        center = feats[idx].mean(axis=0)
        norm = np.linalg.norm(center)
        center = center / norm if norm > 1e-8 else center
        nearest = max(centers, key=lambda key: float(np.dot(center, centers[key])))
        merged[idx] = nearest

    remap = {old: new for new, old in enumerate(sorted(np.unique(merged)))}
    return np.array([remap[item] for item in merged], dtype=int)


def dynamic_cluster_discovery(feats, distance_threshold=None, auto_threshold=True,
                               n_sample_pairs=50000, min_cluster_size=None):
    """
    自底向上层级聚合（Agglomerative Clustering），无需预先指定 K。

    Args:
        feats (np.ndarray): L2-normalized feature embeddings, shape (N, D).
        distance_threshold (float | None): 手动指定余弦距离阈值（1 - 相似度）。
                                           若为 None 且 auto_threshold=True，
                                           则自动从数据分布中估算。
        auto_threshold (bool): 是否开启自动阈值估算。
        n_sample_pairs (int): 自动估算时的采样对数。

    Returns:
        labels (np.ndarray): 每个样本的簇编号。
        centers (dict): 每个簇的 L2-normalized 中心向量。
        num_clusters (int): 最终发现的簇数量。
        used_threshold (float): 本次实际使用的阈值（自动或手动）。
    """
    if len(feats) < 2:
        dummy_center = feats[0] if len(feats) == 1 else np.zeros(feats.shape[1])
        return np.zeros(len(feats), dtype=int), {0: dummy_center}, 1, 0.0

    # 确定使用的阈值
    if distance_threshold is None and auto_threshold:
        used_threshold, _ = estimate_optimal_threshold(feats, n_sample_pairs=n_sample_pairs)
        print(f"   [AutoThreshold] 推断出最优 merge_threshold = {used_threshold:.4f} "
              f"（对应最低相似度 = {1 - used_threshold:.4f}）")
    elif distance_threshold is not None:
        used_threshold = distance_threshold
    else:
        used_threshold = 0.5  # 保底默认值

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=used_threshold,
        metric='cosine',
        linkage='average',
    )
    labels = clustering.fit_predict(feats)
    labels = merge_small_clusters(feats, labels, min_cluster_size=min_cluster_size)

    # 计算各簇中心
    unique_labels = np.unique(labels)
    centers = {}
    for clus_id in unique_labels:
        idx = (labels == clus_id)
        centroid = feats[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        centers[clus_id] = centroid

    return labels, centers, len(unique_labels), used_threshold
