"""
Standalone sanity check, run once by hand (not part of the test suite):
loads both real checkpoints and runs one real inference per language.
If this script fails, the API will fail too -- debug here first, it's
much faster than debugging through HTTP.

Usage:
    source .venv/bin/activate
    python verify_checkpoints.py
"""
from app.model_registry import registry
from app.config import LABELS
import torch

SAMPLES = {
    "en": "Oh sure, because that plan worked out SO well last time.",
    "bn": "তুমি খুব ভালো কাজ করেছ, সত্যিই দারুণ।",
}


def main() -> None:
    print("Loading models...")
    registry.load_all()
    print("Loaded languages:", registry.loaded_languages())

    for lang, text in SAMPLES.items():
        model, tokenizer = registry.get(lang)
        inputs = tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")

        with torch.no_grad():
            logits, hate_logit, sarc_logit = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs).item())

        print(f"\n[{lang}] text: {text}")
        print(f"  predicted label   : {LABELS[label_id]}")
        print(f"  confidence        : {probs[label_id].item():.4f}")
        print(f"  is_hateful_score  : {torch.sigmoid(hate_logit)[0].item():.4f}")
        print(f"  is_sarcastic_score: {torch.sigmoid(sarc_logit)[0].item():.4f}")

    print("\nAll checkpoints loaded and produced predictions successfully.")


if __name__ == "__main__":
    main()
