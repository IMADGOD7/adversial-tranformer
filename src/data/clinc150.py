import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class CLINCDataset(Dataset):
    def __init__(
        self,
        samples: List[List[str]],
        label2id: Dict[str, int],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 32,
    ):
        self.samples = samples
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label_str = self.samples[idx]
        label_id = self.label2id.get(label_str, -1)

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def load_raw_json(raw_path: str) -> Dict[str, List[List[str]]]:
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_and_freeze_label_mapping(
    train_samples: List[List[str]],
    processed_dir: str = "data/processed",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label2id_path = out_dir / "label2id.json"
    id2label_path = out_dir / "id2label.json"

    if label2id_path.is_file() and id2label_path.is_file():
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        with open(id2label_path, "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        return label2id, id2label

    unique_labels = sorted(set(label for _, label in train_samples))
    if len(unique_labels) != 150:
        raise ValueError(f"Expected 150 intents, found {len(unique_labels)}")

    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(id2label_path, "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    return label2id, id2label


def get_dataloaders(
    raw_path: str = "data/raw/data_full.json",
    processed_dir: str = "data/processed",
    backbone_name: str = "distilbert-base-uncased",
    max_seq_len: int = 32,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, DataLoader], Dict[str, int]]:
    raw_data = load_raw_json(raw_path)
    label2id, _ = build_and_freeze_label_mapping(raw_data["train"], processed_dir)
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    loaders = {}
    for split, is_train, bs in [
        ("train", True, batch_size),
        ("val", False, eval_batch_size),
        ("test", False, eval_batch_size),
    ]:
        ds = CLINCDataset(raw_data[split], label2id, tokenizer, max_seq_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    oos_loaders = {}
    for split in ["oos_train", "oos_val", "oos_test"]:
        if split in raw_data:
            ds = CLINCDataset(raw_data[split], label2id, tokenizer, max_seq_len)
            oos_loaders[split] = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)

    return loaders["train"], loaders["val"], loaders["test"], oos_loaders, label2id
