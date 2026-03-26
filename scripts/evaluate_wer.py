import os
from jiwer import wer

PRED_FILE = "results/baseline/predictions.txt"
TRANSCRIPT_DIR = "data/transcripts"


def load_predictions():
    preds = {}
    with open(PRED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            file, text = line.strip().split("\t")
            preds[file] = text
    return preds


def load_ground_truth(filename):
    path = os.path.join(TRANSCRIPT_DIR, filename.replace(".wav", ".txt"))
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    preds = load_predictions()

    refs = []
    hyps = []

    for file, pred in preds.items():
        gt = load_ground_truth(file)

        if gt:
            refs.append(gt)
            hyps.append(pred)

    score = wer(refs, hyps)
    print(f"WER: {score:.4f}")


if __name__ == "__main__":
    main()