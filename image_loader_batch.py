from datasets import load_dataset

repo_id = "saberzl/SID_Set"

train_dataset = load_dataset(repo_id, split="train", streaming = True)
val_dataset = load_dataset(repo_id, split="validation", streaming = True)

for sample in train_dataset:
    print(sample)
