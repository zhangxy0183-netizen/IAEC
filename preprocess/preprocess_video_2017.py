import os
import re
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter

def read_hog_bin(path, num_frames):
    feat_len = 4464
    try:
        data = open(path, 'rb').read()
        offset = 0
        hogs = []
        header_size = 4 * 4
        while offset + header_size + feat_len*4 <= len(data):
            offset += header_size
            vec = np.frombuffer(data, dtype=np.float32, count=feat_len, offset=offset)
            offset += feat_len * 4
            hogs.append(vec)
        hogs = np.stack(hogs, axis=0)
        if hogs.shape[0] != num_frames:
            out = np.zeros((num_frames, feat_len), dtype=np.float32)
            out[:hogs.shape[0]] = hogs
            hogs = out
        return np.nan_to_num(hogs, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        return np.zeros((num_frames, feat_len), dtype=np.float32)

def read_aus_txt(path, num_frames):
    try:
        df = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=0)
        vals = df.iloc[:,4:].values.astype(np.float32)
        out = np.zeros((num_frames, vals.shape[1]), dtype=np.float32)
        out[:vals.shape[0], :vals.shape[1]] = vals[:num_frames]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        return np.zeros((num_frames, 21), dtype=np.float32)

def read_pose_txt(path, num_frames):
    try:
        df = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=0)
        vals = df[['Tx','Ty','Tz','Rx','Ry','Rz']].values.astype(np.float32)
        out = np.zeros((num_frames, 6), dtype=np.float32)
        out[:vals.shape[0], :] = vals[:num_frames]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        return np.zeros((num_frames, 6), dtype=np.float32)

def make_landmark_heatmap(coords, H=64, W=64, sigma=5.0, normalize=True):
    coords = np.nan_to_num(coords, nan=-1.0)
    coords = coords[~(coords[:,0] < 0) & ~(coords[:,1] < 0)]
    if coords.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    coords = np.clip(coords, 0, 224)
    xs = coords[:,0] / 224 * (W-1)
    ys = coords[:,1] / 224 * (H-1)
    heatmap = np.zeros((H, W), dtype=np.float32)
    for x_s, y_s in zip(xs, ys):
        xi, yi = int(round(x_s)), int(round(y_s))
        if 0 <= xi < W and 0 <= yi < H:
            heatmap[yi, xi] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if normalize and heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def process_all_sessions(input_root, output_root, start_id=300, frames_per_clip=15):
    os.makedirs(output_root, exist_ok=True)

    for sess_dir in sorted(os.listdir(input_root)):
        if not sess_dir.endswith("_P"): continue
        sid = sess_dir[:-2]
        try:
            if int(sid) < start_id: continue
        except:
            continue

        in_dir  = os.path.join(input_root, sess_dir)
        out_dir = os.path.join(output_root, sess_dir)
        os.makedirs(out_dir, exist_ok=True)

        img_path = os.path.join(out_dir, f"{sid}_image.npy")
        hm_path  = os.path.join(out_dir, f"{sid}_heatmap.npy")
        if os.path.exists(img_path) and os.path.exists(hm_path):
            print(f"[{sid}] 已处理，跳过。")
            continue

        clnf_path = os.path.join(in_dir, f"{sid}_CLNF_features.txt")
        if not os.path.exists(clnf_path):
            print(f"[{sid}] 缺少 CLNF_features.txt，跳过。")
            continue

        df_clnf = pd.read_csv(clnf_path, sep=r'[\s,]+', engine='python', header=0)
        total_frames = len(df_clnf)
        if total_frames == 0:
            print(f"[{sid}] CLNF_features.txt 为空，跳过。")
            continue
        idxs = np.linspace(0, total_frames-1, frames_per_clip, dtype=int)

        kp_cols = [c for c in df_clnf.columns if re.match(r'^[xX]_?\d+$', c) or re.match(r'^[yY]_?\d+$', c)]
        def sort_key(c):
            num = int(re.findall(r'\d+', c)[0])
            is_y = c.lower().startswith('y')
            return (num, is_y)
        kp_cols = sorted(kp_cols, key=sort_key)
        if len(kp_cols) < 136:
            print(f"[{sid}] 未找到完整 68 点坐标列，跳过。")
            continue

        hog_vals  = read_hog_bin(os.path.join(in_dir, f"{sid}_CLNF_hog.bin"), total_frames)
        aus_vals  = read_aus_txt(os.path.join(in_dir, f"{sid}_CLNF_AUs.csv"), total_frames)
        pose_vals = read_pose_txt(os.path.join(in_dir, f"{sid}_CLNF_pose.txt"), total_frames)

        hogs, aus, poses = [], [], []
        hms = []
        for i in idxs:
            hogs.append(np.nanmean(hog_vals[i]))
            aus.append(np.nanmean(np.abs(aus_vals[i])))
            poses.append(np.linalg.norm(np.nan_to_num(pose_vals[i])))

            vals = df_clnf.iloc[i][kp_cols].astype(float).values
            coords = vals.reshape(-1,2)
            hms.append(make_landmark_heatmap(coords, H=64, W=64, sigma=5.0))

        hogs = np.array(hogs, dtype=np.float32)
        aus  = np.array(aus,  dtype=np.float32)
        poses= np.array(poses,dtype=np.float32)
        def norm(v):
            vmin, vmax = v.min(), v.max()
            return (v - vmin) / (vmax - vmin + 1e-6) if vmax>vmin else np.full_like(v, 0.5)
        hogs_n  = norm(hogs)
        aus_n   = norm(aus)
        poses_n = norm(poses)

        imgs = []
        for t in range(frames_per_clip):
            ch1 = np.full((256,256), hogs_n[t],  dtype=np.float32)
            ch2 = np.full((256,256), aus_n[t],   dtype=np.float32)
            ch3 = np.full((256,256), poses_n[t], dtype=np.float32)
            imgs.append(np.stack([ch1, ch2, ch3], axis=0))

        imgs_arr = np.stack(imgs, axis=0)  # (15,3,256,256)
        hms_arr  = np.stack(hms, axis=0)   # (15,64,64)

        np.save(img_path, np.nan_to_num(imgs_arr))
        np.save(hm_path,  np.nan_to_num(hms_arr))
        print(f"[{sid}] 处理完成：image {imgs_arr.shape}, heatmap {hms_arr.shape}")

if __name__ == "__main__":
    input_root  = "/home/b532root/data/b532zxy/AVEC2017_jieya"
    output_root = "/home/b532root/data/b532zxy/AVEC2017"
    process_all_sessions(input_root, output_root, start_id=396)
