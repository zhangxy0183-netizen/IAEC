import pandas as pd

# 1. 定义三个 CSV 文件的路径
train_csv = "/home/b532root/data/b532zxy/AVEC15/train/train_label.csv"
dev_csv   = "/home/b532root/data/b532zxy/AVEC15/dev/dev_label.csv"
test_csv  = "/home/b532root/data/b532zxy/AVEC15/test/test_label.csv"

# 2. 读取 CSV 文件，将所有 file 列合并到一个列表中
csv_paths = [train_csv, dev_csv, test_csv]
all_files = []
for path in csv_paths:
    df = pd.read_csv(path)
    # 假设 CSV 中的 file 列名为 "file"
    all_files.extend(df["file"].tolist())

# 3. 从每个 file 字符串中提取原始身份编号（例如 "368_1" 提取 "368"），转换为整数方便排序
def extract_identity(file_str):
    try:
        # 按 "_" 分割，取第一个部分，并转为 int
        return int(file_str.split('_')[0])
    except Exception as e:
        raise ValueError(f"无法从文件名 {file_str} 中解析身份信息: {e}")

orig_ids = [extract_identity(f) for f in all_files]

# 4. 去重后按数字大小排序，生成映射字典：原始身份编号 -> 新身份编号（从 0 开始）
unique_ids = sorted(set(orig_ids))
id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}

# 5. 构建结果列表，每个元素包含原始 file 字符串和映射后的 identity
result = []
for f in all_files:
    orig_id = extract_identity(f)
    new_id = id_mapping[orig_id]
    result.append({"file": f, "identity": new_id})

# 6. 将结果保存为 CSV 文件
result_df = pd.DataFrame(result)
result_df.to_csv("/home/b532root/data/b532zxy/AVEC15/file_identity.csv", index=False)

print("已生成 file_identity.csv，其中每个 file 对应的 identity 已映射完成。")