import pandas as pd

train_csv = "/home/b532root/data/b532zxy/AVEC2014/train/train_label.csv"
dev_csv   = "/home/b532root/data/b532zxy/AVEC2014/dev/dev_label.csv"
test_csv  = "/home/b532root/data/b532zxy/AVEC2014/test/test_label.csv"

csv_paths = [train_csv, dev_csv, test_csv]
all_files = []
for path in csv_paths:
    df = pd.read_csv(path)
    all_files.extend(df["file"].tolist())

def extract_identity(file_str):
    try:
        return int(file_str.split('_')[0])
    except Exception as e:
        raise ValueError(f"无法从文件名 {file_str} 中解析身份信息: {e}")

orig_ids = [extract_identity(f) for f in all_files]

unique_ids = sorted(set(orig_ids))
id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}

result = []
for f in all_files:
    orig_id = extract_identity(f)
    new_id = id_mapping[orig_id]
    result.append({"file": f, "identity": new_id})

result_df = pd.DataFrame(result)
result_df.to_csv("/home/b532root/data/b532zxy/AVEC2014/file_identity.csv", index=False)

print("已生成 file_identity.csv，其中每个 file 对应的 identity 已映射完成。")