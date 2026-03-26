# human_like_coding_typing_ai

> A 4-layer AI pipeline that simulates human-like code typing — complete with realistic timing, fatigue, errors, and auto-corrections.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## Overview

`human_like_coding_typing_ai` is the AI engine behind the **vvs** Swift app. It takes a code string as input and produces a sequence of keystroke events that mimic how a real developer would type — including natural pauses at complex blocks, occasional typos, self-corrections, speed variation from fatigue, and bigram-based acceleration.

**Core technique:** HMM (Hidden Markov Model) + GAN (Generative Adversarial Network) + KLM (Keystroke-Level Model) combined to synthesize timing that is statistically indistinguishable from real human input.

---

## Architecture

```
Input Code (string)
        ↓
┌───────────────────────────────────────┐
│  Layer 1 · Code Buffer & Analysis     │  Dependency extraction, language detection
├───────────────────────────────────────┤
│  Layer 2 · AST Scheduling & KLM       │  Tree-sitter AST → complexity classification
│                                       │  → KLM pause injection → nonlinear routing
├───────────────────────────────────────┤
│  Layer 3 · Dynamics Synthesis         │  HMM state sequence → GAN timing samples
│                                       │  → Fitts' Law + bigram speedup + fatigue
│                                       │  → error generation + correction events
├───────────────────────────────────────┤
│  Layer 4 · Injection                  │  Desktop (pyautogui) · Web (Playwright)
│                                       │  · JSON output (Swift subprocess)
└───────────────────────────────────────┘
        ↓
KeystrokeEvent[] + Stats JSON
```

---

## Tech Stack

| Category | Library / Tool | Version | Role |
|----------|---------------|---------|------|
| **Language** | Python | 3.10+ | — |
| **ML Framework** | PyTorch | ≥ 2.2 | GAN 훈련 및 추론 (MPS / CUDA / CPU) |
| **Sequence Model** | hmmlearn | ≥ 0.3 | 타이핑 행동 상태 모델 (6-state HMM) |
| **Statistics** | scipy | ≥ 1.13 | KS test — 생성 분포 vs 실측 분포 검증 |
| **Numerical** | numpy | ≥ 1.26 | 수치 연산 |
| **Code Parsing** | tree-sitter + tree-sitter-languages | ≥ 0.21 | 다중 언어 AST 파싱 (복잡도 분류) |
| **LLM** | Claude API (claude-opus-4-6) | — | 코드 생성 백엔드 |
| **CLI** | click | ≥ 8.1 | 커맨드라인 인터페이스 |
| **Logging** | loguru | ≥ 0.7 | 구조화 로그 |
| **Terminal UI** | rich | ≥ 13.7 | 진행 상황 표시 |
| **Config** | pyyaml | ≥ 6.0 | `config.yaml` 파싱 |
| **Testing** | pytest + pytest-asyncio | ≥ 8.2 | 유닛 / 비동기 테스트 |
| **Desktop Injection** | pyautogui *(optional)* | ≥ 0.9 | 데스크탑 앱 키입력 주입 |
| **Web Injection** | playwright *(optional)* | ≥ 1.44 | 브라우저 키입력 주입 |
| **Data Collection** | pynput *(optional)* | ≥ 1.7 | 실제 키스트로크 데이터 수집 |

---

## Features

- **Multi-language support** — Python, JavaScript, TypeScript, Java, Go, Rust (via Tree-sitter)
- **HMM typing states** — 6 states: NORMAL, SLOW, FAST, ERROR, CORRECTION, PAUSE
- **GAN-based timing** — trained generator produces realistic keydown/keyhold/gap sequences
- **Fitts' Law** — keys farther apart on the keyboard take longer to reach
- **Bigram acceleration** — common character pairs (e.g. `th`, `in`) typed faster
- **Fatigue modeling** — speed gradually decreases over long typing sessions
- **Error simulation** — neighbor key, swap, double, omit errors with auto-correction
- **Complexity-aware pauses** — longer pauses before complex AST nodes (classes, nested loops)
- **Dry-run mode** — simulate without any actual keystrokes
- **Swift subprocess integration** — communicate via stdin/stdout JSON

---

## Installation

```bash
git clone https://github.com/YunMori/human_like_coding_typing_ai.git
cd human_like_coding_typing_ai
pip install -r requirements.txt
```

**Optional dependencies** (install only what you need):

```bash
pip install pyautogui      # Desktop injection
pip install playwright     # Web injection
playwright install chromium

pip install pynput         # Keystroke data collection
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `typing.base_wpm` | 65 | Base typing speed |
| `typing.error_rate` | 0.02 | Probability of a typo per character |
| `typing.fatigue_decay_per_char` | 0.0005 | Speed reduction per character typed |
| `llm.provider` | `claude` | Code generation backend (`claude` / `openai`) |
| `hmm.n_states` | 6 | Number of HMM behavioral states |
| `gan.seq_len` | 32 | Keystroke sequence length for GAN |

---

## Usage

### CLI

```bash
# Type code into a desktop application
python main.py run --lang python --target desktop

# Output timing plan as JSON (for Swift integration)
python main.py type-plan --lang python

# Dry-run — simulate without typing
python main.py run --lang python --target desktop --dry-run

# Inject into a web browser
python main.py run --lang javascript --target web --url http://localhost:3000 --selector "#editor"

# Use custom trained models
python main.py run --lang python --model-dir models/
```

### Swift Subprocess Integration

Send a JSON payload via stdin, receive keystroke events via stdout:

```bash
echo '{"code": "def hello():\n    print(\"world\")", "language": "python"}' \
  | python typing_engine_server.py
```

**Input schema:**
```json
{
  "code": "string",
  "language": "python",
  "model_dir": "models",
  "config_path": "config.yaml",
  "seed": 42
}
```

**Output schema:**
```json
{
  "events": [
    { "key": "d", "delay_before_ms": 120, "key_hold_ms": 80, "is_error": false, "is_correction": false }
  ],
  "stats": {
    "total_keystrokes": 47,
    "error_rate": 0.021,
    "effective_wpm": 61.3,
    "total_duration_ms": 8420
  }
}
```

---

## Training

Pre-trained weights are not included. Training on real human keystroke data is the key step that makes the timing output convincingly human. Without trained models the system falls back to synthetic lognormal distributions, which are noticeably less realistic.

### Step 1 — Prepare training data

**Option A: Use the 2019 CS1 Keystroke Dataset** (recommended)

Download the [2019 CS1 Keystroke Dataset](https://dl.acm.org/doi/10.1145/3287324.3287450) (Utah State University, ~486 students, 5M+ events) and convert it to the JSONL training format:

```bash
python scripts/convert_cs1_dataset.py \
  /path/to/keystrokes.csv \
  --output data/raw/keystroke_samples.jsonl
```

This extracts inter-keystroke intervals, splits sessions at pauses > 5s, and produces 32-keystroke sequences. The resulting dataset (~91K sequences from 486 subjects) has the following timing distribution:

| Percentile | Delay (ms) |
|-----------|------------|
| p25       | 112        |
| p50       | 171        |
| p75       | 363        |
| p95       | 1,701      |

**Option B: Record your own typing**

```bash
pip install pynput
python scripts/collect_keystroke_data.py --duration 300   # 5 min session
```

---

### Step 2 — Train GAN + HMM simultaneously

```bash
# Train both in parallel
PYTHONPATH=. python3 scripts/train_gan.py --epochs 50 &
PYTHONPATH=. python3 scripts/train_hmm.py &
wait
```

Or using the CLI:
```bash
python main.py train-gan   # saves models/gan_generator.pth + gan_discriminator.pth
python main.py train-hmm   # saves models/hmm_model.pkl
```

**GAN training details:**

The GAN uses a bidirectional LSTM Generator + spectral-norm LSTM Discriminator trained with **Hinge loss** (not WGAN-GP, which requires second-order gradients unsupported on Apple MPS). This choice enables native Apple Silicon acceleration via PyTorch MPS.

| Component | Architecture |
|-----------|-------------|
| Generator | LSTM (3 layers, hidden=128) + linear projection → (seq_len, 3) |
| Discriminator | Bidirectional LSTM + spectral norm + hinge loss |
| Optimizer | Adam (lr=1e-4, β=(0.0, 0.9)) |
| D:G update ratio | 2:1 |
| Input | noise (dim=64) + context vector (dim=32) |
| Output | (keydown_ms, keyhold_ms, gap_ms) × seq_len |

The context vector encodes complexity, fatigue level, and HMM state, allowing the generator to condition timing on typing context.

**Hardware acceleration:**

```
Apple Silicon (MPS) → automatically selected
CUDA GPU            → automatically selected if MPS unavailable
CPU                 → fallback
```

Training 50 epochs on 91K sequences takes approximately:
- Apple M-series (MPS): ~1.2 hours
- CPU only: ~8–10 hours

**HMM training details:**

A 6-state `GaussianHMM` (via hmmlearn) is fit on the inter-keystroke delay sequences. The 6 states capture distinct behavioral modes observed in real typing:

| State | Behavior |
|-------|----------|
| NORMAL | Steady comfortable typing |
| SLOW | Thinking before a complex expression |
| FAST | Muscle-memory bursts (common bigrams) |
| ERROR | Pre-error state — slightly irregular |
| CORRECTION | Backspace + retype sequence |
| PAUSE | Deliberate stop (end of block, reading) |

At inference time, the HMM generates a state sequence for each segment, and the GAN samples timing conditioned on those states.

---

### Step 3 — Evaluate

```bash
python scripts/benchmark.py
```

Runs a Kolmogorov-Smirnov test comparing generated timing distributions against held-out real sequences. A KS statistic < 0.1 indicates the generated timing is statistically similar to human typing.

---

## Project Structure

```
.
├── main.py                        # CLI entry point (click)
├── typing_engine_server.py        # Swift subprocess interface
├── config.yaml                    # Model & timing parameters
├── requirements.txt
├── core/
│   ├── pipeline.py                # 4-layer orchestrator
│   └── session.py                 # SessionConfig / SessionResult
├── layer1_codegen/
│   ├── code_buffer.py             # Code + metadata container
│   └── dependency_extractor.py   # Multi-language static analysis
├── layer2_scheduler/
│   ├── scheduler.py               # Layer 2 orchestrator
│   ├── ast_parser.py              # Tree-sitter AST parsing
│   ├── block_classifier.py        # Complexity classification
│   ├── klm_scheduler.py           # KLM pause calculation
│   ├── pause_injector.py          # Pause injection
│   ├── nonlinear_router.py        # Back-visit routing
│   └── typing_plan.py             # TypingPlan / TypingSegment
├── layer3_dynamics/
│   ├── timing_synthesizer.py      # Core synthesis engine
│   ├── hmm_engine.py              # HMM state machine
│   ├── error_generator.py         # Typo simulation
│   ├── correction_engine.py       # Auto-correction events
│   ├── fatigue_model.py           # Speed decay over time
│   ├── bigram_model.py            # Character-pair speedup
│   ├── keyboard_layout.py         # Fitts' Law distance
│   └── gan/                       # GAN model (generator, discriminator, trainer)
├── layer4_injection/
│   ├── desktop_injector.py        # pyautogui injection
│   ├── web_injector.py            # Playwright injection
│   ├── json_output_injector.py    # JSON output
│   └── injector_factory.py
├── data/
│   ├── bigram_frequencies.json    # Pre-computed bigram stats
│   ├── raw/                       # Collected keystroke data
│   └── processed/
├── models/                        # Trained model weights
├── scripts/
│   ├── collect_keystroke_data.py
│   ├── train_gan.py
│   ├── train_hmm.py
│   └── benchmark.py
└── tests/
```

---

## License

MIT
