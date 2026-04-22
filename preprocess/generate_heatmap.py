import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import face_alignment
from scipy.ndimage import gaussian_filter

def generate_combined_heatmap(image_path, output_size=(64, 64), sigma=2):
    # 加载预训练的 FAN 模型，指定 68 个关键点
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    
    # 加载并缩放原始图像到 64x64
    image = io.imread(image_path)
    resized_image = transform.resize(image, output_size, anti_aliasing=True)
    
    # 获取面部关键点
    landmarks = fa.get_landmarks_from_image(image)
    if landmarks is None:
        print(f"No landmarks detected for {image_path}")
        return None, resized_image
    
    # 将原始关键点坐标按比例缩放到 64x64 尺寸
    original_size = image.shape[:2]
    scale_x = output_size[1] / original_size[1]
    scale_y = output_size[0] / original_size[0]
    landmarks_rescaled = [(int(x * scale_x), int(y * scale_y)) for (x, y) in landmarks[0]]
    
    # 创建 64x64 的单个热力图，将所有关键点叠加到一个通道中
    combined_heatmap = np.zeros(output_size, dtype=np.float32)
    for x, y in landmarks_rescaled:
        if 0 <= x < output_size[1] and 0 <= y < output_size[0]:  # 确保坐标在边界内
            combined_heatmap[y, x] = 255  # 设置关键点位置为 1
    combined_heatmap = gaussian_filter(combined_heatmap, sigma=sigma)  # 高斯平滑
    # 归一化到 [0, 1]
    combined_heatmap /= combined_heatmap.max()
    return combined_heatmap, resized_image

def generate_and_save_all_heatmaps(image_paths, save_dir, output_size=(64, 64), sigma=2, alpha=0.6):
    # 检查输出目录是否存在
    os.makedirs(save_dir, exist_ok=True)
    skipped_images = 0
    for image_name in os.listdir(image_paths):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_paths, image_name)
            heatmap, resized_image = generate_combined_heatmap(image_path, output_size=output_size, sigma=sigma)
            if heatmap is not None:
                # 保存 64x64 的热力图
                save_path_npy = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_heatmap.npy")
                np.save(save_path_npy, heatmap)
                # print(f"Saved combined heatmap .npy for {image_path} to {save_path_npy}")

                # 使用颜色映射增强热力图的视觉效果
                heatmap_colored = plt.get_cmap("jet")(heatmap / heatmap.max())[:, :, :3]  # 取 RGB 通道
                heatmap_colored = (heatmap_colored * 255).astype(np.uint8)  # 转换到 0-255 范围
                
                # 叠加热力图到缩放后的图像
                overlay = resized_image * (1 - alpha) + heatmap_colored * alpha
                overlay = np.clip(overlay, 0, 1)  # 限制到 [0, 1] 范围
                
                # # 保存叠加后的图像
                # save_path_img = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_heatmap_overlay.png")
                # plt.imsave(save_path_img, overlay)
                # print(f"Saved overlay image for {image_path} to {save_path_img}")

            else:
                print(f"Skipping {image_path} due to no detected landmarks.")
                skipped_images += 1
            
    print(f"Total images skipped: {skipped_images}")
    print(f"Saved .npy for {save_dir}")
# 示例用法
if __name__ == '__main__':
    image_paths = [
        "/home/b532root/data/b532zxy/AVEC15/face/dev/Freeform",
        "/home/b532root/data/b532zxy/AVEC15/face/dev/Northwind",
        "/home/b532root/data/b532zxy/AVEC15/face/test/Freeform",
        "/home/b532root/data/b532zxy/AVEC15/face/test/Northwind",
        "/home/b532root/data/b532zxy/AVEC15/face/train/Freeform",
        "/home/b532root/data/b532zxy/AVEC15/face/train/Northwind"
    ]

    for directory in image_paths:
        for dir in sorted(os.listdir(directory)):
            if dir not in ["audio_npy"]:

                save_dir = os.path.join(directory, dir, "heatmaps/")
                path = os.path.join(directory, dir)
                generate_and_save_all_heatmaps(path, save_dir)