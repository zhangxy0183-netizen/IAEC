import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import face_alignment
from scipy.ndimage import gaussian_filter

def generate_combined_heatmap(image_path, output_size=(64, 64), sigma=2):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    image = io.imread(image_path)
    resized_image = transform.resize(image, output_size, anti_aliasing=True)
    
    landmarks = fa.get_landmarks_from_image(image)
    if landmarks is None:
        print(f"No landmarks detected for {image_path}")
        return None, resized_image
    
    original_size = image.shape[:2]
    scale_x = output_size[1] / original_size[1]
    scale_y = output_size[0] / original_size[0]
    landmarks_rescaled = [(int(x * scale_x), int(y * scale_y)) for (x, y) in landmarks[0]]
    
    combined_heatmap = np.zeros(output_size, dtype=np.float32)
    for x, y in landmarks_rescaled:
        if 0 <= x < output_size[1] and 0 <= y < output_size[0]:
            combined_heatmap[y, x] = 255
    combined_heatmap = gaussian_filter(combined_heatmap, sigma=sigma)
    combined_heatmap /= combined_heatmap.max()
    return combined_heatmap, resized_image

def generate_and_save_all_heatmaps(image_paths, save_dir, output_size=(64, 64), sigma=2, alpha=0.6):
    os.makedirs(save_dir, exist_ok=True)
    skipped_images = 0
    for image_name in os.listdir(image_paths):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_paths, image_name)
            heatmap, resized_image = generate_combined_heatmap(image_path, output_size=output_size, sigma=sigma)
            if heatmap is not None:
                save_path_npy = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_heatmap.npy")
                np.save(save_path_npy, heatmap)

                heatmap_colored = plt.get_cmap("jet")(heatmap / heatmap.max())[:, :, :3]
                heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
                
                overlay = resized_image * (1 - alpha) + heatmap_colored * alpha
                overlay = np.clip(overlay, 0, 1)

            else:
                print(f"Skipping {image_path} due to no detected landmarks.")
                skipped_images += 1
            
    print(f"Total images skipped: {skipped_images}")
    print(f"Saved .npy for {save_dir}")

if __name__ == '__main__':
    image_paths = [
        "/home/b532root/data/b532zxy/AVEC2014/dev/",
        "/home/b532root/data/b532zxy/AVEC2014/test/",
        "/home/b532root/data/b532zxy/AVEC2014/train/"
    ]

    for directory in image_paths:
        for dir in sorted(os.listdir(directory)):
            if dir not in ["audio_npy", "hubert", "wav2vec2"]:

                save_dir = os.path.join(directory, dir, "heatmaps/")
                path = os.path.join(directory, dir)
                generate_and_save_all_heatmaps(path, save_dir)