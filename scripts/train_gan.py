#!/usr/bin/env python3
"""Standalone GAN training script."""
import click
import yaml
from pathlib import Path


@click.command()
@click.option("--config", default="config.yaml")
@click.option("--data", default="data/raw/keystroke_samples.jsonl")
@click.option("--epochs", default=50)
@click.option("--batch-size", default=64)
@click.option("--output-dir", default="models")
@click.option("--resume", is_flag=True, default=False, help="Resume from best checkpoint")
def main(config, data, epochs, batch_size, output_dir, resume):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    from layer3_dynamics.gan.trainer import GANTrainer
    Path(output_dir).mkdir(exist_ok=True)
    trainer = GANTrainer(cfg.get("gan", {}))
    trainer.train(data, epochs=epochs, batch_size=batch_size, resume=resume)
    trainer.save(f"{output_dir}/gan_generator.pth", f"{output_dir}/gan_discriminator.pth")
    print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    main()
