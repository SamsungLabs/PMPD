# adapted from https://github.com/SafeAILab/EAGLE/blob/d08fe3f23e5f1d986bb50f786af60e5c4f7f757e/eagle/application/webui.py#L4
import os
import time

import gradio as gr
import argparse
import torch
from fastchat.model import get_conversation_template
from pmpd import PMPDForCausalLM, Scheduler
import re


def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]


def find_list_markers(text):

    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)


    return [(match.start(), match.end()) for match in matches]


def checkin(pointer,start,marker):
    for b,e in marker:
        if b<=pointer<e:
            return True
        if b<=start<e:
            return True
    return False


def highlight_text(text, highlight_text, color="orange"):
    result = ""
    
    if text:
        text = torch.stack(text)
        text = text.view(-1)
        text = text.cpu().numpy()
        text = text.tolist()
        text = model.tokenizer.decode(text, skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )
        result += text
    
    if highlight_text:
        highlight_text = torch.stack(highlight_text)
        highlight_text = highlight_text.view(-1)
        highlight_text = highlight_text.cpu().numpy()
        highlight_text = highlight_text.tolist()
        highlight_text = model.tokenizer.decode(highlight_text, skip_special_tokens=True, spaces_between_special_tokens=False,
                                                clean_up_tokenization_spaces=True, )
        result += " "
        chunk = ""
        for char in highlight_text:
            if char != '\n':
                chunk += char
            else:
                if chunk:
                    result += f"<span style='color: {color};'>{chunk}</span>"
                result += char
                chunk = ""
        if chunk:
            result += f"<span style='color: {color};'>{chunk}</span>"
    return result


def warmup():
    conv = get_conversation_template('vicuna')
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    set_scheduler(model, precisions, "Static", high_bit_steps=0)
    for output_ids in model.pmpd_generate(input_ids):
        _=output_ids


def bot(history, model_choice, scheduler_choice, highlight_pmpd, classifier_path, high_bit_steps, session_state):
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    conv = get_conversation_template('vicuna')

    for query, response in pure_history:
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], response)

    prompt = conv.get_prompt()

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    low_prec_text = []
    high_prec_text = []
    precs = []
    cu_len = input_len
    totaltime=0
    start_time=time.time()
    total_ids=0
    
    # set scheduler
    set_scheduler(model, precisions, scheduler_choice, classifier_path, high_bit_steps)
    if model_choice == 'pmpd':

        for output_ids, current_bit in model.pmpd_generate(input_ids, 
                                             max_new_tokens=args.max_new_token):
            totaltime+=(time.time()-start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            # prec_text = model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
            #                                          spaces_between_special_tokens=False,
            #                                          clean_up_tokenization_spaces=True, )
            if current_bit < max(model.scheduler.precisions):
              low_prec_text.append(output_ids[0, cu_len])
            else:
              high_prec_text.append(output_ids[0, cu_len])
            cu_len = output_ids.shape[1]
            colored_text = highlight_text(high_prec_text, low_prec_text, "orange")
            if highlight_pmpd:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cu_len-input_len
            precs.append(current_bit)
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{sum(precs)/len(precs):.2f}",session_state
            start_time = time.time()


    else:
        if model_choice == 'High Precision':
            precision = max(model.scheduler.precisions)
        else: 
            precision = min(model.scheduler.precisions)
        for output_ids in model.naive_generate(input_ids, precision, max_new_tokens=args.max_new_token):
            totaltime += (time.time() - start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            cu_len = output_ids.shape[1]
            history[-1][1] = text
            pure_history[-1][1] = text
            new_tokens = cu_len - input_len
            precs.append(precision)
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{sum(precs)/len(precs):.2f}",session_state
            start_time = time.time()
            

def user(user_message, history,session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]],session_state


def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state


def set_scheduler(model, precisions, scheduler, classifier_path=None, high_bit_steps=None):
  torch.cuda.empty_cache()  
  kw_dict = {}
  kw_dict['precisions'] = precisions
  # argument for naive scheduler
  kw_dict['high_bit_steps'] = int(high_bit_steps)
  # argument for kv_cache scheduler
  precision_switch_points = set()
  for p in sorted(precisions, reverse=True):
      if p-1 in precisions:
          precision_switch_points.add((p, p-1))
  kw_dict['precision_switch_points'] = precision_switch_points
  kw_dict['save_dir'] = classifier_path
  config = model.model.config
  kw_dict['dim'] = config.hidden_size // config.num_attention_heads
  kw_dict['num_heads'] = config.num_key_value_heads
  kw_dict['max_new_tokens'] = args.max_new_token
  
  scheduler = 'naive' if scheduler == 'Static' else 'kv_cache'
  model.scheduler = Scheduler.get_scheduler(scheduler, **kw_dict)


def update_fields(scheduler):
    if scheduler == "Learned":
        return gr.update(interactive=True), gr.update(interactive=False)
    else:
        return gr.update(interactive=False), gr.update(interactive=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    default="anyprec-vicuna-7b-4-2/",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
  "--precisions",
  type=str,
  default="3,2",
  help="The precisions to use for the model.",
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=256,
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

precisions = [int(p) for p in args.precisions.split(",")]
model = PMPDForCausalLM(
  args.model_path,
  precisions=precisions,
  use_anyprec=True,
)
model.cuda()
model.eval()
warmup()

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": [], "model": model})
    gr.Markdown('''## pmpd Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        bitwidth_box = gr.Textbox(label="Average Bitwidth", elem_id="speed", interactive=False, value="0.00")

    chatbot = gr.Chatbot(height=600,show_label=False)


    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        clear_button = gr.Button("Clear")
    
    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(choices=["pmpd", "High Precision", "Low Precision"], label="Model", value="pmpd")
            highlight_pmpd = gr.Checkbox(label="Highlight the tokens generated by Low Precision Model", value=True)
        with gr.Column():
            scheduler_choice = gr.Radio(choices=["Static", "Learned"], label="Scheduler", value="Static")
            classifier_path = gr.Textbox(label="Classifier Path", value="test/anyprec-vicuna-7b-4-2_3_2_256/", interactive=False)
            high_bit_steps = gr.Textbox(label="High Bit Steps", value="10", interactive=True)
        # temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0.5)
    note=gr.Markdown(show_label=False,value='''''')
    enter_event=msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, model_choice, scheduler_choice, highlight_pmpd, classifier_path, high_bit_steps, gs], [chatbot,speed_box,bitwidth_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,bitwidth_box,gs], queue=True)

    send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
        bot, [chatbot, model_choice, scheduler_choice, highlight_pmpd, classifier_path, high_bit_steps, gs], [chatbot,speed_box,bitwidth_box,gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,enter_event])
    
    scheduler_choice.change(update_fields, [scheduler_choice], [classifier_path, high_bit_steps])
demo.queue()
demo.launch(share=True)