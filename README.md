# Qwen3 ASR Shanghainese Experiment

This project explores adapting Qwen3-ASR for Shanghainese speech recognition.

## Structure

- `data/`: audio + transcripts
- `scripts/`: training / evaluation
- `results/`: outputs
- `notebooks/`: analysis

## Workflow

### Step 1: Baseline Test
Run zero-shot ASR on Shanghainese data.

```bash
python scripts/baseline_test.py