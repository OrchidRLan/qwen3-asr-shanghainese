import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio

DATA_DIR = "data/processed"
OUTPUT_FILE = "results/baseline/predictions.txt"

MODEL_NAME = "Qwen/Qwen3-ASR"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to(device)


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze()


def transcribe(audio_path):
    audio = load_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def main():
    os.makedirs("results/baseline", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for file in os.listdir(DATA_DIR):
            if file.endswith(".wav"):
                path = os.path.join(DATA_DIR, file)
                print(f"Processing: {file}")

                pred = transcribe(path)

                f.write(f"{file}\t{pred}\n")

    print("Done. Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()