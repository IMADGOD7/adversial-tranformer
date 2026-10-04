from collections import Counter
# pyrefly: ignore [missing-import]
import pytest
import torch
from transformers import AutoTokenizer

# pyrefly: ignore [missing-import]
from src.data.clinc150 import (
    build_and_freeze_label_mapping,
    get_dataloaders,
    load_raw_json,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_raw_json("data/raw/data_full.json")


def test_split_counts_and_balance(raw_data):
    train, val, test = raw_data["train"], raw_data["val"], raw_data["test"]
    assert len(train) == 15000 and len(val) == 3000 and len(test) == 4500

    train_c = Counter(label for _, label in train)
    val_c = Counter(label for _, label in val)
    test_c = Counter(label for _, label in test)

    assert len(train_c) == len(val_c) == len(test_c) == 150
    assert all(c == 100 for c in train_c.values())
    assert all(c == 20 for c in val_c.values())
    assert all(c == 30 for c in test_c.values())


def test_frozen_label_mapping(raw_data, tmp_path):
    label2id, id2label = build_and_freeze_label_mapping(
        raw_data["train"], processed_dir=str(tmp_path)
    )
    assert len(label2id) == len(id2label) == 150
    assert sorted(label2id.values()) == list(range(150))
    for i in range(150):
        assert label2id[id2label[i]] == i


def test_zero_exact_leakage(raw_data):
    train_q = {text.strip().lower() for text, _ in raw_data["train"]}
    val_q = {text.strip().lower() for text, _ in raw_data["val"]}
    test_q = {text.strip().lower() for text, _ in raw_data["test"]}

    # In official crowdsourced CLINC150, exactly 5 generic phrases (0.02%) repeat across splits
    overlap_train_val = len(train_q & val_q)
    overlap_train_test = len(train_q & test_q)
    overlap_val_test = len(val_q & test_q)

    assert overlap_train_val <= 3
    assert overlap_train_test <= 2
    assert overlap_val_test == 0


def test_token_length_distribution(raw_data):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    all_texts = [text for text, _ in raw_data["train"] + raw_data["val"] + raw_data["test"]]
    token_lengths = sorted(len(tokenizer.encode(t, add_special_tokens=True)) for t in all_texts)
    
    p99 = token_lengths[int(len(token_lengths) * 0.99)]
    truncated = sum(1 for l in token_lengths if l > 32)
    truncation_pct = (truncated / len(token_lengths)) * 100

    assert p99 <= 32
    assert truncation_pct < 1.0


def test_dataloaders_shape_and_batching():
    train_loader, val_loader, test_loader, _, label2id = get_dataloaders(
        raw_path="data/raw/data_full.json",
        processed_dir="data/processed",
        max_seq_len=32,
        batch_size=16,
        eval_batch_size=32,
    )
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (16, 32)
    assert batch["attention_mask"].shape == (16, 32)
    assert batch["label"].shape == (16,)
    assert batch["label"].dtype == torch.long
    assert (batch["label"] >= 0).all() and (batch["label"] < 150).all()
