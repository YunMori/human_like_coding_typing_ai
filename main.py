#!/usr/bin/env python3
"""VVS Typing AI - Real coding simulation auto-typing system."""
import asyncio
import json
import sys
from pathlib import Path

import click
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


@click.group()
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


@cli.command()
@click.argument("prompt")
@click.option("--lang", "-l", default="python", help="Programming language")
@click.option("--target", "-t", default="desktop", type=click.Choice(["desktop", "web", "json_output"]), help="Injection target")
@click.option("--url", default=None, help="URL for web target")
@click.option("--selector", "-s", default=None, help="CSS selector for web target")
@click.option("--dry-run", is_flag=True, help="Simulate without injecting")
@click.option("--model-dir", default="models", help="Directory with trained model weights")
@click.pass_context
def run(ctx, prompt, lang, target, url, selector, dry_run, model_dir):
    """Run the typing AI pipeline."""
    config = ctx.obj["config"]

    console.print(Panel(
        f"[bold cyan]VVS Typing AI[/bold cyan]\n"
        f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}\n"
        f"Language: {lang} | Target: {target} | Dry run: {dry_run}",
        title="Starting Pipeline"
    ))

    from core.pipeline import TypingPipeline
    from core.session import SessionConfig

    session = SessionConfig(
        prompt=prompt,
        language=lang,
        target=target,
        url=url,
        selector=selector,
        dry_run=dry_run,
    )

    pipeline = TypingPipeline(config, model_dir=model_dir)

    async def _run():
        return await pipeline.run(session)

    result = asyncio.run(_run())

    table = Table(title="Session Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Session ID", result.session_id)
    table.add_row("Language", result.language)
    table.add_row("Total Keystrokes", str(result.total_keystrokes))
    table.add_row("Duration", f"{result.total_duration_ms/1000:.1f}s")
    table.add_row("Error Rate", f"{result.error_rate:.1%}")
    table.add_row("Corrections", str(result.correction_count))
    table.add_row("Avg WPM", f"{result.avg_wpm:.0f}")
    console.print(table)


@cli.command(name="type-plan")
@click.option("--code", required=True, help="Code to generate typing plan for")
@click.option("--lang", "-l", default="python", help="Programming language")
@click.option("--model-dir", default="models", help="Directory with trained model weights")
@click.option("--seed", default=None, type=int, help="Random seed for reproducible timing")
@click.pass_context
def type_plan(ctx, code, lang, model_dir, seed):
    """Generate a JSON timing plan for the given code (Swift vvs integration)."""
    config = ctx.obj["config"]

    from core.pipeline import TypingPipeline
    pipeline = TypingPipeline(config, model_dir=model_dir)
    result = pipeline.generate_timing_plan(code, lang, seed=seed)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


@cli.command()
@click.option("--epochs", default=50, help="Training epochs")
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--data", default="data/raw/keystroke_samples.jsonl", help="Training data path")
@click.option("--output-dir", default="models", help="Output directory for models")
@click.pass_context
def train_gan(ctx, epochs, batch_size, data, output_dir):
    """Train the GAN timing model."""
    config = ctx.obj["config"]
    from layer3_dynamics.gan.trainer import GANTrainer
    Path(output_dir).mkdir(exist_ok=True)
    trainer = GANTrainer(config.get("gan", {}))
    trainer.train(data, epochs=epochs, batch_size=batch_size)
    trainer.save(f"{output_dir}/gan_generator.pth", f"{output_dir}/gan_discriminator.pth")
    console.print("[green]GAN training complete![/green]")


@cli.command()
@click.option("--output-dir", default="models", help="Output directory for model")
@click.pass_context
def train_hmm(ctx, output_dir):
    """Train the HMM model from keystroke data."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train_hmm.py", "--output-dir", output_dir],
        capture_output=True, text=True
    )
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"[red]{result.stderr}[/red]")


@cli.command()
@click.option("--output", default="data/raw/keystroke_samples.jsonl", help="Output file")
@click.option("--duration", default=300, help="Recording duration in seconds")
def collect(output, duration):
    """Collect keystroke data for training."""
    import subprocess
    subprocess.run(
        [sys.executable, "scripts/collect_keystroke_data.py", "--output", output, "--duration", str(duration)],
    )


@cli.command()
@click.option("--model-dir", default="models", help="Directory with trained models")
@click.pass_context
def benchmark(ctx, model_dir):
    """Benchmark GAN quality using KS test."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--model-dir", model_dir],
        capture_output=True, text=True
    )
    console.print(result.stdout)


if __name__ == "__main__":
    cli()
