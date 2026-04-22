#!/usr/bin/env python3
import os
import shutil

def main():
    # w_all wo_id wo_sim wo_se
    mode = 'w_all'
    # 定义源文件路径和目标重命名后的路径
    src_cam = "/home/b532root/data/b532zxy/AVEC15/weights_cam/best.pth"
    src_feature = "/home/b532root/data/b532zxy/AVEC15/weights_feature/best.pth"
    src_mask = "/home/b532root/data/b532zxy/AVEC15/weights_mask/mask.pth"

    new_name_cam = "/home/b532root/data/b532zxy/AVEC15/weights_cam/best_cam_" + mode + ".pth"
    new_name_feature = "/home/b532root/data/b532zxy/AVEC15/weights_feature/best_fea_" + mode + ".pth"
    new_name_mask = "/home/b532root/data/b532zxy/AVEC15/weights_mask/best_mask" + mode + ".pth"

    destination = "/home/b532root/data/b532zxy/AVEC15/result"
    
    dest_cam = os.path.join(destination, "best_cam_" + mode + ".pth")
    dest_feature = os.path.join(destination, "best_fea_" + mode + ".pth")
    dest_mask = os.path.join(destination, "best_mask_" + mode + ".pth")
    # 定义目标目录

    
    # 确保目标目录存在
    os.makedirs(destination, exist_ok=True)

    # 1. 重命名源文件（避免移动时重名冲突）
    try:
        print(f"重命名 {src_cam} 为 {new_name_cam}")
        os.rename(src_cam, new_name_cam)
    except Exception as e:
        print(f"重命名 {src_cam} 失败: {e}")

    try:
        print(f"重命名 {src_feature} 为 {new_name_feature}")
        os.rename(src_feature, new_name_feature)
    except Exception as e:
        print(f"重命名 {src_feature} 失败: {e}")

    try:
        print(f"重命名 {src_mask} 为 {new_name_mask}")
        os.rename(src_mask, new_name_mask)
    except Exception as e:
        print(f"重命名 {src_mask} 失败: {e}")

    try:
        print(f"移动 {new_name_cam} 到 {dest_cam}")
        shutil.move(new_name_cam, dest_cam)
    except Exception as e:
        print(f"移动 {new_name_cam} 失败: {e}")

    try:
        print(f"移动 {new_name_feature} 到 {dest_feature}")
        shutil.move(new_name_feature, dest_feature)
    except Exception as e:
        print(f"移动 {new_name_feature} 失败: {e}")

    try:
        print(f"移动 {new_name_mask} 到 {dest_mask}")
        shutil.move(new_name_mask, dest_mask)
    except Exception as e:
        print(f"移动 {new_name_mask} 失败: {e}")

    print("文件移动完成！")

if __name__ == "__main__":
    main()
