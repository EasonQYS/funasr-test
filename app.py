import gradio as gr
import os,sqlite3,shutil
from modelscope import snapshot_download
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

print(os.system('ls -la /home/xlab-app-center/.cache/'))
print(os.system('ls -la /home/xlab-app-center/.cache/modelscope'))

local_path = '/home/xlab-app-center/.cache/modelscope/hub/iic/SenseVoiceSmall'
if os.path.exists(local_path):
    print("文件已存在:", local_path)
else:
    model_dir = snapshot_download('iic/SenseVoiceSmall')

model = AutoModel(
    model=local_path,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

def do(a):
    #print(a.name)
    path = "./" + os.path.basename(a.name)  # 复制到新目录
    shutil.copyfile(a.name, path)
    res = model.generate(
        input=path,
        cache={},
        language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size=64, 
    )
    text = rich_transcription_postprocess(res[0]["text"])
    #print(text)
    return text
    
def do2(a):
    print(a)
    path = "./" + os.path.basename(a.name)  # 复制到新目录
    shutil.copyfile(a.name, path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    filename = '/home/xlab-app-center/data.json'
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
    return filename
    
with gr.Blocks() as demo:
    gr.Markdown('# SNH48语料生成器')
    gr.Markdown('## Part 1: 直播语音转文本')
    gr.Markdown('第一步：B站视频转MP3。可以通过第三方平台实现，如[这个](https://www.xtdowner.com/video/bilibili/)')
    gr.Markdown('第二步：上传MP3之后，转文本，通过[阿里FunASR](https://github.com/modelscope/FunASR)实现。')
    with gr.Row():
        a = gr.File(label='待转换的mp3',file_types=['.mp3'])
        t = gr.Textbox(label='转换文本的结果')
    b = gr.Button("立即转换")
    gr.Markdown('## Part 2：口袋48聊天数据转json')
    gr.Markdown('聊天数据可以用[这个StarBot](https://github.com/Jaffoo/StarBot)抓取，文件在`./wwwroot/data/data.db`中，上传这个文件，可以转换为XTuner模型微调的数据。')
    with gr.Row():
        f1 = gr.File(file_types=['.db'],label='数据库聊天记录')
        b2 = gr.Button("数据库转json")
        f2 = gr.File(label='数据语料下载')
    file_path = gr.FileExplorer(root_dir='./')
    b.click(do, inputs=a,outputs=t)
    b2.click(do2, inputs=f1,outputs=f2)
demo.launch()
