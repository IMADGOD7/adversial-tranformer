import pytest
import torch
from torch.optim import AdamW

# pyrefly: ignore [missing-import]
from src.models.multiexit_distilbert import MultiExitDistilBERT


@pytest.fixture(scope="module")
def model():
    m = MultiExitDistilBERT(num_classes=150)
    m.eval()
    return m


def test_forward_shapes(model):
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.tensor([10, 42], dtype=torch.long)

    out = model(input_ids, attention_mask, labels=labels)

    assert "loss" in out and out["loss"].ndim == 0
    assert set(out["logits"].keys()) == {"exit_2", "exit_4", "exit_6"}
    assert set(out["representations"].keys()) == {"exit_2", "exit_4", "exit_6"}

    for exit_name in ["exit_2", "exit_4", "exit_6"]:
        assert out["logits"][exit_name].shape == (2, 150)
        assert out["representations"][exit_name].shape == (2, 768)


def test_early_exit_equivalence(model):
    input_ids = torch.randint(0, 1000, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        full_out = model(input_ids, attention_mask)
        logits_2, _ = model.forward_early_exit(input_ids, attention_mask, stop_at_exit=2)
        logits_4, _ = model.forward_early_exit(input_ids, attention_mask, stop_at_exit=4)
        logits_6, _ = model.forward_early_exit(input_ids, attention_mask, stop_at_exit=6)

    assert torch.allclose(full_out["logits"]["exit_2"], logits_2, atol=1e-4)
    assert torch.allclose(full_out["logits"]["exit_4"], logits_4, atol=1e-4)
    assert torch.allclose(full_out["logits"]["exit_6"], logits_6, atol=1e-4)


def test_overfit_small_batch():
    m = MultiExitDistilBERT(num_classes=150)
    m.train()
    optimizer = AdamW(m.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)
    labels = torch.tensor([5, 25, 75, 120], dtype=torch.long)

    initial_loss = m(input_ids, attention_mask, labels=labels)["loss"].item()

    for _ in range(15):
        optimizer.zero_grad()
        loss = m(input_ids, attention_mask, labels=labels)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = m(input_ids, attention_mask, labels=labels)["loss"].item()
    assert final_loss < initial_loss * 0.7
