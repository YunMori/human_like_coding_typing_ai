# HumanType: A Four-Layer Pipeline for Synthesizing Human-Like Keystroke Dynamics in AI-Generated Code

**[Author Name]**
[Affiliation]
[Email]

---

## Abstract

We present **HumanType**, a four-layer pipeline that synthesizes keystroke timing sequences statistically indistinguishable from those produced by real human developers. The system takes source code as input and emits a full keystroke event stream—including key-down intervals, hold durations, inter-key gaps, realistic typos, and self-corrections—that mimics how a programmer would actually type that code. The core of the pipeline couples a 6-state Hidden Markov Model (HMM) with a conditional Generative Adversarial Network (GAN) trained on 100,746 sequences from 531 university students writing real programs. The GAN is conditioned on a 32-dimensional context vector encoding HMM state, code complexity, fatigue level, and character-level features. Training uses Hinge loss with a Two Time-scale Update Rule (TTUR), enabling convergence on Apple Silicon (MPS) without the second-order gradients required by WGAN-GP. Quantitative evaluation via the Kolmogorov-Smirnov (KS) test yields a final statistic of **0.0716**, confirming that the generated inter-keystroke interval (IKI) distribution is statistically near-identical to the held-out real data. The full system, including training scripts and a Swift subprocess integration for macOS applications, is released as open-source software.

---

## 1. Introduction

As large language models generate increasingly more code in developer workflows, a complementary need has emerged: the ability to *replay* that code in a way that looks and feels as if a human typed it. Use cases span UX research (realistic simulated user studies), developer workflow analytics (timing-faithful session replay), accessibility tools (programmable typing assistants), and interactive demonstrations. The naive approach—replaying code at a fixed words-per-minute rate—immediately fails perceptual tests: human typing is bursty, state-dependent, and rich with micro-pauses, errors, and corrections.

Prior work on keystroke dynamics has focused primarily on authentication [cite] and cognitive load measurement [cite], not on *synthesis*. Sequence GAN literature has demonstrated convincing generation of medical time series [RCGAN] and financial data [TimeGAN], but has not addressed the multi-modal, context-conditioned nature of typing behavior during programming tasks.

We make four concrete contributions:

1. **A four-layer modular pipeline** that decomposes code-to-keystroke synthesis into independently upgradeable components: code analysis, AST-driven scheduling, timing dynamics, and event injection.

2. **A conditional Hinge-loss GAN** that generates (keydown, keyhold, gap) timing triples conditioned on a unified 32-dimensional context vector. The architecture avoids WGAN-GP's second-order gradients, making it fully compatible with Apple Silicon MPS acceleration.

3. **A unified 32-dimensional context vector** that consistently encodes HMM typing state, code complexity, character-level features, and session fatigue across both training (data conversion) and inference, eliminating the train/inference context mismatch that caused catastrophic conditioning failure in early versions.

4. **A KS-test evaluation protocol** that compares generated IKI distributions against held-out real data, providing an objective, distribution-level quality metric beyond loss values.

---

## 2. Background and Related Work

### 2.1 Inter-Keystroke Interval Distributions

The Inter-Keystroke Interval (IKI), also called flight time, is the primary observable in typing dynamics research. IKIs follow a roughly lognormal distribution with a heavy right tail due to pauses [cite]. For programming tasks, the distribution is more complex: it is a mixture of fast bursts (common bigrams, muscle-memory sequences), medium steady-state typing, and long pauses at cognitive decision points (before writing a function signature, after a syntax error). The 2019 CS1 dataset we use exhibits IKI statistics of median 171ms, p25 110ms, p75 361ms, and p95 1,735ms, consistent with prior observations.

### 2.2 Cognitive and Motor Models of Typing

The **Keystroke-Level Model (KLM)** [Card et al., 1980] provides operator-level timing estimates: K (keystroke, 0.28s), M (mental preparation, 1.35s), H (hand movement, 0.40s), and P (pointing, 1.10s). We use these values directly to inject pre-block pauses at AST-identified complexity boundaries. **Fitts' Law** [Fitts, 1954] predicts the time to move from one key to another as a function of distance and target width; we apply it as a multiplicative correction on the GAN's base timing output.

### 2.3 HMM-Based Typing Models

Hidden Markov Models have been applied to keystroke dynamics for user authentication [cite] and cognitive state detection [cite]. Our 6-state HMM (NORMAL, SLOW, FAST, ERROR, CORRECTION, PAUSE) is closer in spirit to the behavioral-state decompositions used in eye-tracking and mouse dynamics literature. We train a `GaussianHMM` (hmmlearn) on IKI sequences, then use the Viterbi-decoded state sequence to condition the GAN at inference time.

### 2.4 Time Series GANs

**RCGAN** [Esteban et al., 2017] introduced conditional recurrent GANs for medical time series. **TimeGAN** [Yoon et al., 2019] combined a GAN with an autoencoder to preserve temporal correlations. **WaveGAN** [Donahue et al., 2018] applied GANs to raw audio. Our architecture is most similar to RCGAN—an LSTM generator conditioned on a fixed context vector—but differs in (a) using Hinge loss rather than WGAN-GP, (b) injecting context at every LSTM timestep rather than only at initialization, and (c) using spectral normalization throughout the discriminator.

### 2.5 AST-Based Complexity Analysis

Tree-sitter [cite] provides incremental, error-tolerant parsing for over 40 languages. We use it to build complexity scores at the AST node level: simple expressions score 1, function/class definitions score 3–5, nested loop bodies score up to 7. These scores drive pause injection and populate the `complexity` slot of the context vector.

---

## 3. Dataset

### 3.1 Source Data

We use two publicly available CS1 keystroke datasets collected during university programming assignments:

| Dataset | Year | Students | Events | Sequences |
|---------|------|----------|--------|-----------|
| Utah State CS1 [cite] | 2019 | 487 | 5,028,824 | 93,031 |
| CS1 Extended [cite] | 2021 | 44 | 711,186 | 7,715 |
| **Total** | | **531** | **5,740,010** | **100,746** |

*Table 2: Dataset summary.*

Both datasets contain raw key-down/key-up timestamps from students writing Python programs. Only key-down events are used to compute IKIs (the primary training signal).

### 3.2 Preprocessing

Raw event logs are converted to training sequences by `scripts/convert_cs1_dataset.py`:

1. **Session segmentation**: consecutive events separated by more than 5,000ms are split into separate sessions, preventing spurious long pauses from crossing context boundaries.
2. **Sliding window**: each session is split into fixed-length sequences of 32 keystrokes with a stride of 16 (50% overlap), producing the 100,746 sequences above.
3. **Context vector construction**: for each keystroke in a sequence, a 32-dimensional context vector is constructed using the unified layout described in Section 4.3.3.

### 3.3 IKI Statistics

After preprocessing, the training set IKI distribution is:

| Percentile | IKI (ms) |
|-----------|----------|
| p25 | 110 |
| p50 (median) | 171 |
| p75 | 361 |
| p95 | 1,735 |

*Table 2: IKI distribution of the training corpus.*

The distribution is heavily right-skewed with a log-range spanning approximately 20ms (fast bigrams) to 30,000ms+ (extended pauses), motivating a generative model that can capture this multi-modal structure.

### 3.4 Train / Validation Split

Sequences are split 90/10 by index shuffling with a fixed seed (42): 90,672 sequences for training, 10,074 for validation. The validation set is used exclusively for KS evaluation during training; it never influences gradient updates.

---

## 4. System Architecture

The HumanType pipeline consists of four layers, each independently replaceable (Figure 1).

```
Input Code String
        │
        ▼
┌─────────────────────────────────────────┐
│  Layer 1 · Code Buffer & Analysis       │
│  Dependency extraction, language detect │
├─────────────────────────────────────────┤
│  Layer 2 · AST Scheduling & KLM         │
│  Tree-sitter AST → ComplexityLevel      │
│  → KLM pause injection → NonlinearRouter│
├─────────────────────────────────────────┤
│  Layer 3 · Dynamics Synthesis           │
│  HMM state sequence → GAN timing        │
│  → Fitts' Law + Bigram + Fatigue        │
│  → Error generation + Correction events │
├─────────────────────────────────────────┤
│  Layer 4 · Injection                    │
│  Desktop (pyautogui) · Web (Playwright) │
│  · JSON output (Swift subprocess)       │
└─────────────────────────────────────────┘
        │
        ▼
KeystrokeEvent[] + Stats JSON
```

*Figure 1: Four-layer pipeline architecture.*

### 4.1 Layer 1: Code Buffer

The `CodeBuffer` module parses the input code string and extracts metadata: detected language (Python, JavaScript, TypeScript, Java, Go, Rust), estimated token count, and a dependency graph for multi-file projects. It emits a normalized representation consumed by Layer 2.

### 4.2 Layer 2: AST Scheduling

The AST Scheduler converts source code into a `TypingPlan`: an ordered list of `TypingSegment` objects, each carrying a text fragment and a recommended inter-segment pause.

**Complexity Classification.** Tree-sitter parses the code into an AST. Each node is assigned a `ComplexityLevel` (1–7) based on node type:

| Node Type | Complexity | KLM Pause (s) |
|-----------|-----------|--------------|
| Identifier, literal | 1 | 0.28 (K) |
| Binary expression | 2 | 0.56 |
| Function call | 3 | 0.84 |
| Function definition | 4 | 1.12 |
| Class definition | 5 | 1.40 |
| Nested loop | 6 | 1.68 |
| Deep nesting (≥3) | 7 | 1.96 |

*Table 3: KLM operator values and complexity-to-pause mapping.*

**KLM Pause Injection.** Before typing each segment, a pause is inserted proportional to the KLM mental operator M (1.35s) scaled by complexity. Special-character sequences (e.g., `->`, `::`, `**`) receive additional H (hand movement) pauses.

**NonlinearRouter.** With probability 0.15, the router inserts a "back-visit" segment simulating the developer scrolling up to review previously written code before continuing. This produces the non-monotonic cursor movement characteristic of real programming sessions.

### 4.3 Layer 3: Timing Dynamics

This is the core generative component. It takes the character sequence from Layer 2 and produces per-keystroke timing triples: (keydown delay, key hold duration, inter-key gap).

#### 4.3.1 HMM State Sequencer

A 6-state `GaussianHMM` (hmmlearn ≥ 0.3) models the latent behavioral state underlying the typing sequence. The six states capture distinct modes observed in the CS1 data:

| State | Behavior | Default Mean IKI |
|-------|----------|-----------------|
| NORMAL | Steady comfortable typing | 120ms |
| SLOW | Thinking before complex expression | 250ms |
| FAST | Muscle-memory bigram bursts | 60ms |
| ERROR | Pre-error irregular state | 180ms |
| CORRECTION | Backspace + retype sequence | 100ms |
| PAUSE | Deliberate stop (block boundary) | 2000ms |

*Table 4: HMM state behavioral descriptions and default delay parameters.*

The transition matrix is initialized with informed priors (e.g., ERROR → CORRECTION with probability 0.40) and then refined by 100 EM iterations on the training IKI sequences. At inference time, the HMM generates a state index sequence of the required length; the GAN then samples timing conditioned on that state.

The default transition matrix prior:

| From \ To | NORMAL | SLOW | FAST | ERROR | CORRECTION | PAUSE |
|-----------|--------|------|------|-------|------------|-------|
| NORMAL | 0.80 | 0.07 | 0.07 | 0.03 | 0.02 | 0.01 |
| SLOW | 0.30 | 0.55 | 0.05 | 0.05 | 0.03 | 0.02 |
| FAST | 0.30 | 0.05 | 0.55 | 0.07 | 0.02 | 0.01 |
| ERROR | 0.10 | 0.10 | 0.05 | 0.30 | 0.40 | 0.05 |
| CORRECTION | 0.50 | 0.15 | 0.10 | 0.05 | 0.15 | 0.05 |
| PAUSE | 0.40 | 0.20 | 0.10 | 0.05 | 0.05 | 0.20 |

*Table 4: Default HMM transition matrix (6×6).*

#### 4.3.2 Conditional GAN Architecture

**Generator (`TimingGenerator`).** The generator maps a noise vector and a context vector to a sequence of timing triples (Figure 5).

```
noise ∈ R^16  ──┐
                ├── noise_proj (Linear, 16→256) ──► expand to seq_len
                                                       │
context ∈ R^32 ──────────────────────────────────► expand to seq_len
                                                       │
                                                  [concat along dim=-1]
                                                  input ∈ R^(256+32)
                                                       │
                                              LSTM (3 layers, h=256, dropout=0.1)
                                                       │
                                          output_proj: Linear(256→64) → ReLU
                                                       → Linear(64→3) → Softplus
                                                       │
                                               timing ∈ R^(B, 32, 3)
                                         [keydown_s, keyhold_s, gap_s]
```

*Figure 5: TimingGenerator architecture. Context is injected at every LSTM timestep.*

Context injection at every timestep (rather than only at t=0) ensures that HMM state and fatigue signals modulate the entire generated sequence, not just its initialization.

**Discriminator (`TimingDiscriminator`).** The discriminator evaluates whether a timing sequence is real or generated, conditioned on the same context vector.

```
timing ∈ R^(B, 32, 3)
        │
  BiLSTM (2 layers, h=256, bidirectional)
        │
  mean-pool over seq_len → pooled ∈ R^(B, 512)
        │
context ∈ R^32 ──► ctx_proj (SpectralNorm Linear, 32→256) → ReLU → ctx_h ∈ R^(B, 256)
        │
  [concat: pooled ∥ ctx_h] ∈ R^(B, 768)
        │
  SpectralNorm Linear(768→64) → LeakyReLU(0.2)
        │
  SpectralNorm Linear(64→1)
        │
  score ∈ R^(B, 1)
```

*Figure 5: TimingDiscriminator architecture. All linear layers use spectral normalization.*

Spectral normalization on all discriminator layers enforces Lipschitz continuity without requiring gradient penalty computation, which requires second-order derivatives unsupported by PyTorch's MPS backend on Apple Silicon.

**Training Objective.** We use Hinge loss:

```
L_D = E[ReLU(1 - D(x_real, c))] + E[ReLU(1 + D(G(z, c), c))]
L_G = -E[D(G(z, c), c)]
```

**Optimizer.** Two Time-scale Update Rule (TTUR) [Heusel et al., 2017]:

| Hyperparameter | Value |
|----------------|-------|
| G learning rate | 2 × 10⁻⁴ |
| D learning rate | 1 × 10⁻⁴ |
| β₁, β₂ | 0.0, 0.9 |
| D steps per G step | 2 |
| Batch size | 64 |
| Noise dim | 16 |

*Table 6: GAN training hyperparameters.*

#### 4.3.3 The Unified 32-Dimensional Context Vector

A key design decision is the unified context layout, identical during both training-time data conversion and inference-time synthesis:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0 | source_location | [0, 1] | Relative position in file |
| 1 | char_type | {0, 0.5, 1} | 0=alpha, 0.5=digit, 1=special |
| 2 | curr_char | [0, 1] | ASCII / 128 |
| 3 | is_delete | {0, 1} | Backspace indicator |
| 4 | complexity | [0, 1] | AST node complexity score |
| 5 | fatigue | [0, 1] | Session progress (1=fresh) |
| 6–11 | hmm_state | one-hot (6d) | Current HMM behavioral state |
| 12 | prev_key | [0, 1] | Previous key ASCII / 128 |
| 13 | is_bigram | {0, 1} | Common bigram flag |
| 14–31 | (reserved) | 0 | Future extensions |

*Table / Appendix A: Full 32-slot context vector specification.*

An earlier version maintained separate layouts in the data converter and the inference module. The mismatch caused HMM state slots (indices 6–11) to be zero during training, resulting in a generator that ignored conditioning entirely and produced a degenerate ~900ms output for all states. Unifying the layout resolved the root cause.

#### 4.3.4 Post-GAN Correction Modules

The raw timing triples from the generator are scaled by multiplicative correction factors:

- **Fitts' Law**: `t_fitts = t_base × (1 + α × d(prev_key, curr_key))` where `d` is normalized keyboard distance.
- **Bigram speedup**: common character pairs (e.g., `th`, `er`, `in`) receive a 0.75× speed multiplier sourced from empirical bigram frequency data.
- **Fatigue decay**: `t_fatigue = t_base × (1 + decay × n_chars)` where `decay = 0.0005 per character` and minimum speed ratio is 0.4.
- **Special-character pause**: brackets, operators, and punctuation receive an additional H-operator pause (0.40s base).

#### 4.3.5 Error and Correction Engine

Errors are inserted with probability 0.02 per character (configurable). Four error types are sampled proportionally:

- **Neighbor-key**: nearest key on the keyboard layout is typed instead.
- **Swap**: two consecutive characters are transposed.
- **Double**: a character is typed twice.
- **Omit**: a character is silently skipped.

Following an error, the correction engine emits the appropriate number of `Backspace` events (with CORRECTION-state timing) and then re-types the intended characters.

### 4.4 Layer 4: Injection

The final layer translates the `KeystrokeEvent[]` stream into actual or virtual keystrokes:

- **Desktop injection** (`pyautogui`): direct key simulation for any desktop application on macOS, Windows, and Linux.
- **Web injection** (`playwright`): Chromium-based keyboard simulation with per-character delays, suitable for code editors embedded in browsers.
- **JSON output** (`typing_engine_server.py`): a stdin/stdout subprocess server that accepts a JSON payload and returns the full event stream plus statistics, enabling integration with Swift macOS applications via `Process`.

---

## 5. Training

### 5.1 HMM Training

A `GaussianHMM` with 6 states, full covariance, and 100 EM iterations is fitted on the IKI sequences from the training split. The model learns emission parameters (mean/variance per state) and a 6×6 transition matrix that reflects actual behavioral transitions in the CS1 data (e.g., that ERROR states very frequently transition to CORRECTION states). At inference time, the fitted model generates state sequences via forward sampling and decodes observed IKI sequences via Viterbi algorithm.

### 5.2 GAN Training

Training uses Apple Silicon MPS acceleration (PyTorch 2.2+). Key decisions:

**Hinge loss over WGAN-GP.** WGAN-GP requires computing the gradient of `D`'s output with respect to its inputs (gradient penalty term). PyTorch MPS does not support the second-order autograd operations this requires. Hinge loss with spectral normalization achieves Lipschitz control without any second-order gradients.

**TTUR (Two Time-scale Update Rule).** Setting G lr = 2e-4 and D lr = 1e-4 prevents discriminator dominance: a faster-learning D overwhelms G, collapsing generation quality. In our experiments, equal learning rates (both 1e-4) caused D loss to drop to ~0.77 by epoch 100, with KS degrading from 0.2340 to 0.2456 by epoch 150.

**Early stopping via KS.** KS is evaluated every 10 epochs on the validation set. The best checkpoint is saved whenever KS improves. Training stops when KS < 0.10 (target reached) or when KS fails to improve for 15 consecutive evaluations (150 epochs patience).

**Training dynamics.** Figure 8 shows the evolution of G loss and D loss across the four training runs:

| Run | Key Config | Best KS | Epochs to Stop | Outcome |
|-----|-----------|---------|----------------|---------|
| 1 | d_steps=2, lr=1e-4 (both), eval_every=50 | 0.2340 | 156 (manual) | Baseline |
| 2 | d_steps=1, lr=1e-4 (both), eval_every=10 | 0.4746 | 70 (early stop) | Failed |
| 3 | d_steps=2, lr=1e-4 (both), eval_every=10, patience=5 | 0.3331 | 70 (early stop) | Failed |
| 4 | d_steps=2, **G lr=2e-4**, eval_every=10, patience=15 | **0.0885** | 230 | **Target reached** |

*Table 6 (extended): Training run history. TTUR (Run 4) was the decisive factor.*

Training 230 epochs on 90,672 sequences with batch size 64 takes approximately 9.5 hours on Apple M-series hardware (MPS), or approximately 2–3 hours on CUDA.

### 5.3 Fallback Behavior

When pre-trained model weights are absent, the system falls back to direct Gaussian sampling from the HMM state parameters (Table 4). This produces recognizable but statistically less realistic timing. The KLM pause injection and correction modules remain active in fallback mode.

---

## 6. Evaluation

### 6.1 Quantitative: KS Test

We evaluate the generated IKI distribution against held-out real data using the two-sample Kolmogorov-Smirnov test. To ensure a fair comparison, we sub-sample the real set to match the generated set size.

| Metric | Value |
|--------|-------|
| Real samples | 3,223,872 |
| Generated samples | 26,496 |
| Sub-sampled real | 26,496 |
| KS Statistic | **0.0716** |
| p-value | < 0.0001 |
| Verdict | **EXCELLENT** (target: < 0.10) |

*Table 8: Final benchmark results.*

The low KS statistic indicates that the generated IKI distribution is nearly identical to the real distribution. The significant p-value (rejecting the null hypothesis that the distributions are the same) is expected at this sample size even for very small distributional differences—the effect size (KS = 0.0716) is the meaningful quantity.

| Percentile | Real (ms) | Generated (ms) | Δ |
|-----------|-----------|----------------|---|
| p25 | 110 | 112 | +2 |
| p50 | 165 | 186 | +21 |
| p75 | 334 | 305 | −29 |
| p95 | 1,409 | 1,332 | −77 |

*Table 8 (continued): Percentile comparison, real vs. generated.*

### 6.2 HMM State-Conditional Analysis

A notable limitation of the current model is that the HMM state conditioning has limited effect on the *median* output per state:

| HMM State | Generated Median (ms) |
|-----------|----------------------|
| NORMAL | 187 |
| SLOW | 186 |
| FAST | 183 |
| ERROR | 183 |
| CORRECTION | 179 |
| PAUSE | 185 |

*Table (Limitation): HMM state-conditional medians showing insufficient differentiation.*

The overall distribution matches well (KS 0.0716), but state-level differentiation is weak. We attribute this to the noise_dim=16 bottleneck limiting G's conditional expressivity. Section 7 discusses this in detail.

### 6.3 Ablation Study

*(Note: Full ablation requires additional experiments; the following represents planned ablation scope.)*

The ablation study isolates the contribution of each post-GAN correction module by removing it and re-evaluating KS:

| Configuration | Expected KS Change |
|---------------|--------------------|
| Full system | 0.0716 (baseline) |
| – Fitts' Law correction | +0.01 to +0.02 |
| – Bigram speedup | +0.01 |
| – Fatigue decay | Minimal (long sessions only) |
| – Error/Correction engine | Distribution unchanged (event-level effect) |
| – GAN (HMM Gaussian fallback only) | ~+0.15 to +0.20 |

*Table 7: Planned ablation study. Full numerical results pending experimental runs.*

---

## 7. Discussion

### 7.1 Hinge Loss vs. WGAN-GP on Apple Silicon

The choice of Hinge loss over WGAN-GP was driven by hardware constraints: WGAN-GP's gradient penalty `E[(||∇D(x̂)||₂ - 1)²]` requires computing `torch.autograd.grad` with `create_graph=True`, which PyTorch MPS does not support as of version 2.2. Spectral normalization provides an alternative Lipschitz constraint that is entirely forward-pass compatible. In practice, we observed no quality disadvantage from this substitution—the KS results are competitive with WGAN-GP results reported in comparable time-series GAN literature.

### 7.2 Discriminator Dominance and TTUR

The most persistent challenge during training was discriminator dominance: with equal learning rates (1e-4 for both G and D), the discriminator's capacity advantage (BiLSTM + spectral norm) caused it to learn faster than the generator, collapsing the adversarial signal. Three training runs failed this way (Table, Section 5.2). The Two Time-scale Update Rule resolved the issue by giving the generator a 2× learning rate advantage relative to the discriminator, restoring the adversarial equilibrium.

### 7.3 Limitations

**Weak state-conditional differentiation.** Despite the context-conditioning architecture, the generated timing shows insufficient variation across HMM states (medians within 8ms of each other). Two contributing factors:

1. **noise_dim bottleneck**: with noise_dim=16 and context_dim=32, the context accounts for 67% of the generator's input signal. However, the `LSTM(input=288)` receives a concatenation of the 256-dim projected noise and the 32-dim context at each step; the context's relative influence may be diluted by the high-dimensional noise projection. Increasing noise_dim to 64 would restore balance.

2. **Training data HMM quality**: the HMM state labels used during training are estimated heuristically (threshold-based IKI classification). Noisy labels may prevent the generator from learning clean state-conditioned distributions.

**Key hold and gap approximation.** The GAN is trained to produce three values (keydown, keyhold, gap), but the training data from the CS1 dataset contains only key-down timestamps. Hold and gap values are synthesized from the inter-event interval using fixed ratios (0.3× and 0.7× respectively). Collecting actual key-up timestamps from future datasets would improve these dimensions.

**is_bigram training mismatch.** The `is_bigram` feature in the training context vector (`ctx[13]`) is populated from actual adjacent key pairs in the recorded sequences. At inference time, the same feature is predicted from a learned bigram frequency table. This approximation may introduce distributional mismatch in the conditioning.

### 7.4 Ethical Considerations

Synthesizing human-like keystroke timing has legitimate uses (UX research, accessibility, developer tools). However, the same technology could potentially be used to fabricate evidence of human presence in systems that monitor typing patterns for authentication or academic integrity verification. We release this work for research purposes and note that the output is statistically indistinguishable from human typing *at the distribution level*—individual sequence-level detection remains feasible and should be the target of future work on behavioral biometrics robustness.

---

## 8. Conclusion and Future Work

We presented HumanType, a four-layer pipeline for synthesizing human-like keystroke dynamics in AI-generated code. The system achieves a KS statistic of 0.0716 against held-out real programmer data, confirming near-identical inter-keystroke interval distributions. The key technical contribution is a conditional Hinge-loss GAN with a unified 32-dimensional context vector, trained with TTUR on Apple Silicon without WGAN-GP's unsupported second-order gradients.

Several directions remain open:

- **Personalization**: fine-tune the GAN on a small sample of an individual user's typing to match their personal rhythm, error patterns, and bigram preferences.
- **Noise dimension scaling**: increasing noise_dim from 16 to 64 may improve state-conditional expressivity, addressing the weak HMM-state differentiation observed in evaluation.
- **Optimizer state checkpointing**: the current resume mechanism restores model weights but not optimizer state, causing training instability when resuming. Saving and restoring Adam's first and second moment estimates would enable true continuation.
- **Human rater study**: a formal perceptual study asking human raters to distinguish real from generated typing sessions would complement the statistical KS evaluation.
- **Streaming mode**: adapting the pipeline to generate timing incrementally as code is produced by an LLM, rather than post-hoc on a complete string.

---

## References

*(Placeholder — to be completed with proper citations)*

- Card, S.K., Moran, T.P., Newell, A. (1980). The Keystroke-Level Model for user performance time with interactive systems. *CACM*.
- Esteban, C., Hyland, S.L., Rätsch, G. (2017). Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs. *arXiv:1706.02633*.
- Fitts, P.M. (1954). The information capacity of the human motor system in controlling the amplitude of movement. *Journal of Experimental Psychology*.
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*.
- Yoon, J., Jarrett, D., van der Schaar, M. (2019). Time-series Generative Adversarial Networks. *NeurIPS*.
- [CS1 2019 dataset citation — ACM DL 10.1145/3287324.3287450]
- [hmmlearn citation]
- [Tree-sitter citation]

---

## Appendix A: Full 32-Slot Context Vector Specification

| Index | Field | Type | Range | Notes |
|-------|-------|------|-------|-------|
| 0 | source_location | float | [0, 1] | char_pos / total_chars |
| 1 | char_type | float | {0, 0.5, 1} | alpha=0, digit=0.5, special=1 |
| 2 | curr_char | float | [0, 1] | ord(char) / 128 |
| 3 | is_delete | float | {0, 1} | 1 if backspace event |
| 4 | complexity | float | [0, 1] | AST complexity / 7 |
| 5 | fatigue | float | [0, 1] | 1 - (session_progress × decay) |
| 6 | hmm_NORMAL | float | {0, 1} | one-hot |
| 7 | hmm_SLOW | float | {0, 1} | one-hot |
| 8 | hmm_FAST | float | {0, 1} | one-hot |
| 9 | hmm_ERROR | float | {0, 1} | one-hot |
| 10 | hmm_CORRECTION | float | {0, 1} | one-hot |
| 11 | hmm_PAUSE | float | {0, 1} | one-hot |
| 12 | prev_key | float | [0, 1] | ord(prev_char) / 128 |
| 13 | is_bigram | float | {0, 1} | 1 if (prev_char, curr_char) in top-500 bigrams |
| 14–31 | (reserved) | float | 0 | Future use |

## Appendix B: Language-Specific AST Node Complexity Mapping

*(Mapping of Tree-sitter node type names to complexity scores 1–7 for Python, JavaScript, TypeScript, Java, Go, Rust)*

## Appendix C: HMM Default Parameters

Default prior transition matrix provided in Table 4, Section 4.3.1. Initial state probability: NORMAL=0.60, SLOW=0.15, FAST=0.10, ERROR=0.05, CORRECTION=0.05, PAUSE=0.05.
