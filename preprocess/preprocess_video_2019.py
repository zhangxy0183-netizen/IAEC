import os
import numpy as np
import pandas as pd
import cv2

def make_aus_heatmap(au_vals, H=64, W=64, sigma=5.0):
    positions = [(i // 5 * (H//4), i % 5 * (W//5)) for i in range(len(au_vals))]
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    hm = np.zeros((H, W), dtype=np.float32)
    for (r, c), val in zip(positions, au_vals):
        hm += val * np.exp(-((xx - c)**2 + (yy - r)**2) / (2 * sigma * sigma))
    if hm.max() > 0:
        hm /= hm.max()
    return hm

def load_openface_csv(csv_path):
    header_idx = None
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            low = line.lower()
            if 'frame' in low and ('gaze' in low or '_r' in low):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"无法在 {csv_path!r} 中找到表头行")
    return pd.read_csv(csv_path, header=header_idx)

def process_e_daic(input_root, output_root, frames_per_clip=15):
    os.makedirs(output_root, exist_ok=True)

    for sess in sorted(os.listdir(input_root)):
        if not sess.endswith("_P"):
            continue

        sid      = sess[:-2]  # '300_P' -> '300'
        feat_dir = os.path.join(input_root, sess, "features")
        out_dir  = os.path.join(output_root, sess)
        os.makedirs(out_dir, exist_ok=True)

        # 只匹配真正的 OpenFace CSV
        candidates = []
        for fname in os.listdir(feat_dir):
            low = fname.lower()
            if (low.startswith(f"{sid.lower()}_openface")
                and 'pose' in low
                and 'gaze' in low
                and 'au' in low
                and low.endswith('.csv')):
                candidates.append(fname)
        if not candidates:
            print(f"Skip {sess}: 无 OpenFace CSV")
            continue

        csv_name = candidates[0]
        csv_path = os.path.join(feat_dir, csv_name)
        print(f"[{sess}] 使用 CSV → {csv_name}")

        df = load_openface_csv(csv_path)
        N  = len(df)
        if N == 0:
            print("  → CSV 为空，跳过")
            continue
        idxs = np.linspace(0, N-1, frames_per_clip, dtype=int)

        cols      = df.columns.tolist()
        gaze_cols = [c for c in cols if 'gaze' in c.lower()][:3]
        pose_cols = [c for c in cols if 'pose' in c.lower()][:3]
        au_cols   = [c for c in cols if c.endswith('_r')]
        if not (gaze_cols and pose_cols and au_cols):
            print("  → 缺少期望列，跳过")
            continue

        # 第一步：收集每帧各通道标量
        vals1, vals2, vals3 = [], [], []
        hms = []
        for i in idxs:
            row = df.iloc[i]
            val1 = np.abs(row[au_cols].astype(float).values).mean()
            val2 = np.linalg.norm(row[gaze_cols].astype(float).values)
            val3 = np.linalg.norm(row[pose_cols].astype(float).values)
            vals1.append(val1)
            vals2.append(val2)
            vals3.append(val3)
            # heatmap
            hm64 = make_aus_heatmap(row[au_cols].astype(float).values, H=64, W=64)
            hms.append(hm64)

        vals1 = np.array(vals1); vals2 = np.array(vals2); vals3 = np.array(vals3)
        # 通道归一化参数
        def norm_array(v):
            vmin, vmax = v.min(), v.max()
            if vmax - vmin < 1e-6:
                return np.full_like(v, 0.5, dtype=np.float32)
            return (v - vmin) / (vmax - vmin)
        norm1 = norm_array(vals1)
        norm2 = norm_array(vals2)
        norm3 = norm_array(vals3)

        # 第二步：生成伪图像和上采样热力图
        imgs, hms256 = [], []
        for t in range(frames_per_clip):
            ch1 = np.full((256,256), norm1[t], dtype=np.float32)
            ch2 = np.full((256,256), norm2[t], dtype=np.float32)
            ch3 = np.full((256,256), norm3[t], dtype=np.float32)
            imgs.append(np.stack([ch1, ch2, ch3], axis=0))
            hms256.append(cv2.resize(hms[t], (256,256), interpolation=cv2.INTER_CUBIC))

        imgs_arr = np.stack(imgs, axis=0)   # (15,3,256,256)
        hms_arr  = np.stack(hms256, axis=0) # (15,256,256)

        np.save(os.path.join(out_dir, f"{sess}_image.npy"),   imgs_arr)
        np.save(os.path.join(out_dir, f"{sess}_heatmap.npy"), hms_arr)
        print(f"  → 保存 image {imgs_arr.shape}, heatmap {hms_arr.shape}")

if __name__ == "__main__":
    process_e_daic(
        input_root  = "/home/b532root/data/b532zxy/AVEC2019_jieya",
        output_root = "/home/b532root/data/b532zxy/AVEC2019"
    )
