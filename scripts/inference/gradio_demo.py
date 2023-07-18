import visualcla
from visualcla.modeling_utils import DEFAULT_GENERATION_CONFIG
import argparse
import os
import gradio as gr
import mdtex2html
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--visualcla_model', default=None, type=str, required=True,
                    help="Path to the merged VisualCLA model")
parser.add_argument('--gpus', default="0", type=str,
                    help="GPU(s) to use for inference")
parser.add_argument('--share', default=False, action='store_true',
                    help='share gradio domain name')
parser.add_argument('--load_in_8bit',action='store_true',
                    help="Whether to load the LLM in 8bit (only supports merged VisualCLA model)")
parser.add_argument('--only_cpu',action='store_true',
                    help='Only use CPU for inference')
parser.add_argument('--no_stream',action='store_true',
                    help='Output without stream mode.')
args = parser.parse_args()
share = args.share
load_in_8bit = args.load_in_8bit
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model = None

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

# Borrowed from VisualGLM (https://github.com/THUDM/VisualGLM-6B)
def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input_text, image_path_upload, image_path_webcam, chatbot, max_new_tokens, top_p, top_k, temperature, history, selected):
    if selected=='Upload':
        image_path = image_path_upload
        print("Image from upload: ", image_path_upload)
    elif selected=='Webcam':
        image_path = image_path_webcam
        print("Image from webcam: ", image_path_webcam)
    else:
        raise ValueError(selected)
    DEFAULT_GENERATION_CONFIG.top_p = top_p
    DEFAULT_GENERATION_CONFIG.top_k = top_k
    DEFAULT_GENERATION_CONFIG.max_new_tokens = max_new_tokens
    DEFAULT_GENERATION_CONFIG.temperature = temperature
    if image_path is None:
        return [(input_text, "图片不能为空。请重新上传图片。")], []
    chatbot.append((parse_text(input_text), ""))
    with torch.no_grad():
        if args.no_stream:
            response, history = visualcla.chat(model, image=image_path, text=input_text, history=history, generation_config=DEFAULT_GENERATION_CONFIG)
            chatbot[-1] = (parse_text(input_text), parse_text(response))
            yield chatbot, history
        else:
            strean_generator = visualcla.chat_in_stream(model, image=image_path, text=input_text, history=history, generation_config=DEFAULT_GENERATION_CONFIG)
            for response, history in strean_generator:
                chatbot[-1] = (parse_text(input_text), parse_text(response))
                yield chatbot, history

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return None, [], []


def main():
    global model
    print("Loading the model...")
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
        device_map='auto'
    else:
        device = torch.device('cpu')
        device_map={'':device}
    load_in_8bit = args.load_in_8bit
    base_model, tokenizer, _ = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model=args.visualcla_model,
        torch_dtype=load_type,
        default_device=device,
        device_map=device_map,
        load_in_8bit=load_in_8bit and (args.visualcla_model is not None)
    )
    model = base_model

    if device == torch.device('cpu'):
        model.float()
    model.eval()

    with gr.Blocks(theme=gr.themes.Default()) as demo:

        selected_state = gr.State("Upload")
        def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
            return evt.value
    
        github_banner_path = 'https://raw.githubusercontent.com/airaria/Visual-Chinese-LLaMA-Alpaca/main/pics/banner.png'
        gr.HTML(f'<p align="center"><a href="https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca"><img src={github_banner_path} width="700"/></a></p>')
        with gr.Row():
            with gr.Column(scale=3.5):
                chatbot = gr.Chatbot().style(height=400)
                user_input = gr.Textbox(show_label=False, placeholder="Your Instruction here", lines=4).style(
                    container=False)
                with gr.Row():
                        submitBtn = gr.Button("提交", variant="primary")
                        emptyBtn = gr.Button("清除")
            with gr.Column(scale=2.5):
                    with gr.Tab("Upload") as t1:
                        image_path_upload = gr.Image(type="pil", label="Image", value=None).style(height=310)
                        t1.select(on_select,outputs=selected_state)
                    with gr.Tab("Webcam") as t2:
                        image_path_webcam = gr.Image(type="pil", label="Image", value=None, source='webcam')
                        t2.select(on_select, outputs=selected_state)
                    max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Max new tokens", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.9, step=0.01, label="Top P", interactive=True)
                    top_k = gr.Slider(0, 100, value=40, step=1, label="Top K", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.5, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(predict, [user_input, image_path_upload, image_path_webcam, chatbot, max_new_tokens, top_p, top_k, temperature, history, selected_state], [chatbot, history],
                        show_progress=True)
        image_path_upload.clear(reset_state, outputs=[image_path_upload, chatbot, history], show_progress=True)
        image_path_webcam.clear(reset_state, outputs=[image_path_webcam, chatbot, history], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(lambda: (None, None, [], []), outputs=[image_path_upload, image_path_webcam, chatbot, history], show_progress=True)

        print(gr.__version__)

        demo.queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=8090)

if __name__ == '__main__':
    main()