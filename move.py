import os
import shutil


def safe_move(src, dst):
    """
    安全移动文件：
    1. 源文件不存在：跳过，不报错
    2. 目标文件已存在：先删除旧文件
    """
    if not os.path.exists(src):
        print(f"Skipped: {src} does not exist")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst):
        os.remove(dst)
        print(f"Deleted existing target: {dst}")

    shutil.move(src, dst)
    print(f"Moved and renamed: {src} -> {dst}")


def safe_delete(path):
    """
    安全删除文件：
    文件不存在也不报错
    """
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")
    else:
        print(f"Skipped delete: {path} does not exist")


def rename_and_move_files(base_dir):
    result_dir = os.path.join(base_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    special_masks = [
        "weights_mask_ID_1_SE_1",
        "weights_mask_ID_1_SE_0",
        "weights_mask_ID_0_SE_1",
        "weights_mask_ID_0_SE_0"
    ]

    for folder in special_masks:
        src = os.path.join(base_dir, folder, "mask.pth")
        dst = os.path.join(result_dir, f"{folder}.pth")
        safe_move(src, dst)

    other_folders = [
        "weights_feature_ID_1_SE_1_SIM_1",
        "weights_feature_ID_1_SE_1_SIM_0",
        "weights_feature_ID_1_SE_0_SIM_1",
        "weights_feature_ID_0_SE_1_SIM_1",
        "weights_cam_ID_1_SE_1_SIM_1_VIDEO_1",
        "weights_cam_ID_1_SE_1_SIM_1_VIDEO_0",
        "weights_cam_ID_1_SE_1_SIM_0_VIDEO_1",
        "weights_cam_ID_1_SE_0_SIM_1_VIDEO_1",
        "weights_cam_ID_0_SE_1_SIM_1_VIDEO_1"
    ]

    for folder in other_folders:
        checkpoint_path = os.path.join(base_dir, folder, "checkpoint.pth")
        safe_delete(checkpoint_path)

        src = os.path.join(base_dir, folder, "best.pth")
        dst = os.path.join(result_dir, f"{folder}.pth")
        safe_move(src, dst)


if __name__ == '__main__':
    base_folder_name = 'AVEC2014'
    base_dir = os.path.join("/home/b532root/data/b532zxy", base_folder_name)
    rename_and_move_files(base_dir)