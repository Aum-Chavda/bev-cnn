import torch
from src.utils.config import BEVConfig
from src.models.resnet import BEVResNet
from src.models.registry import ModelRegistry
from src.data.dataset import build_dataloader
from src.training.trainer import Trainer
from src.training.callbacks import PrintLogger, EarlyStopping
from src.utils.visualize import FeatureMapHook, ModelSummary, save_feature_maps


def main():
    # ── config ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BEVConfig(
    epochs=5,
    batch_size=32,
    lr=1e-4,   # lowered from 1e-3
    device=device,
    checkpoint_dir="checkpoints/",
)
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    # ── model ───────────────────────────────────────────────────────────
    model = ModelRegistry.get("bevresnet")(cfg)
    print(ModelSummary(model).summary())

    # ── data ────────────────────────────────────────────────────────────
    train_loader = build_dataloader(split="train", batch_size=cfg.batch_size)
    val_loader   = build_dataloader(split="val",   batch_size=cfg.batch_size)

    # ── train ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.lr,
        epochs=cfg.epochs,
        device=cfg.device,
        checkpoint_dir=cfg.checkpoint_dir,
        callbacks=[
            PrintLogger(),
            EarlyStopping(patience=3),
        ],
    )
    trainer.fit()

    # ── visualize feature maps ──────────────────────────────────────────
    print("\nExtracting feature maps...")
    model.eval()
    sample, _ = next(iter(val_loader))
    sample = sample.to(device)

    with FeatureMapHook(model, ["stage1", "stage2", "stage3", "stage4"]) as hook:
        model(sample)

    save_feature_maps(hook.features, output_dir="outputs/")
    print("\nDone. Check outputs/ for feature map report.")


if __name__ == "__main__":
    main()