import gradio as gr
#SDK模型下载
from modelscope import snapshot_download
local_path = '/home/xlab-app-center/.cache/modelscope/hub/iic/SenseVoiceSmall'
if os.path.exists(local_path):
    print("文件已存在:", local_path)
else:
    model_dir = snapshot_download('iic/SenseVoiceSmall')
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

def do(a):
    print(a)
    return a
    res = model.generate(
        input=a,
        cache={},
        language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size=64, 
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)
    return text

with gr.Blocks() as demo:
    with gr.Row():
        a = gr.Audio(label='待转换的mp3',type='filepath')
        t = gr.Textbox(label='转换文本的结果')
    b = gr.Button("立即转换")
    b.click(do, inputs=a,outputs=t)
    
demo.launch()
