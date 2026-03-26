# test_shanghainese.py
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# 加载模型和处理器
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "./Qwen3-ASR-1.7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./Qwen3-ASR-1.7B")


import soundfile as sf

# 读取音频文件
audio_input, sample_rate = sf.read("./data/raw/shanghainese_test_baba.WAV")

# 处理音频输入
inputs = processor(
    audio_input,
    sampling_rate=sample_rate,
    return_tensors="pt"
)

# 进行识别
with torch.no_grad():
    outputs = model.generate(**inputs)

# 解码识别结果
transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"识别结果: {transcription}")


import gradio as gr

def recognize_speech(audio_file):
    # 音频处理和识别代码
    # ...
    return transcription

# 创建界面
iface = gr.Interface(
    fn=recognize_speech,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="方言语音识别演示"
)

iface.launch()







# from qwen_asr import Qwen3ASRModel
# import torch

# model = Qwen3ASRModel.from_pretrained(
#     "./Qwen3-ASR-1.7B",   # 本地路径
#     dtype=torch.bfloat16,
#     device_map="cuda:0",
# )

# result = model.transcribe(
#     "data/raw/shanghainese_test_baba.WAV",
#     language="Wu language"   # 或者试试 "wuu"（吴语 ISO 639-3 代码）
# )


# print("原始返回值：", result)
# print("类型：", type(result))