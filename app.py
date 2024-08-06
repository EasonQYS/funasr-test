import gradio as gr
import os,sqlite3
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
def do2(a):
    print(a)
    conn = sqlite3.connect(a)
    cursor = conn.cursor()
    filename = 'data.json'
    # 执行SQL查询
    query = "SELECT * FROM 'PoketMessage' "
    cursor.execute(query)

    # 获取查询结果
    rows = cursor.fetchall()
    num = len(rows)
    print(len(rows))
    print(rows[len(rows)-1])
    with open(filename,'a',encoding='utf-8') as f:
        f.write('[')
    # 打印结果
    last_msg = ''
    for row in rows:
        msg = row[1]
        msg = msg.replace('\n','->')
        msg = msg.replace('\n','->')
        #print(msg[:6])
        if msg[:5]=='https' or msg[:5]=='/2024' or msg[0]=='[':
            continue
        #print(row[1])
        with open(filename,'a',encoding='utf-8') as f:
            f.write('{"conversation":[{\n')
            f.write('  "input": "'+last_msg+'",\n')
            f.write('  "output": "'+msg+'"\n')
            f.write('}]},\n')
        last_msg = msg

    with open(filename,'a',encoding='utf-8') as f:
        f.write(']')
    # 关闭Cursor
    cursor.close()

    # 关闭连接
    conn.close()
    return './data.json'
    
with gr.Blocks() as demo:
    with gr.Row():
        a = gr.Audio(label='待转换的mp3',type='filepath')
        t = gr.Textbox(label='转换文本的结果')
    b = gr.Button("立即转换")
    with gr.Row():
        f1 = gr.File(file_types=['.db'])
        b2 = gr.Button("立即转换")
        f2 = gr.File(file_types=['.json'])
    b.click(do, inputs=a,outputs=t)
    b2.click(do2, inputs=f1,outputs=f2)
    
    
demo.launch()
