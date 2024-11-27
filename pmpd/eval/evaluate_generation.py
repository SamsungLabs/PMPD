"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import transformers

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

import sys

from human_eval.data import write_jsonl, read_problems
from datasets import load_dataset
from any_precision import AnyPrecisionForCausalLM
from pmpd import PMPDForCausalLM, Scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM


def prompt_CNN(article):

    prompt="""

For the following article: {} 

Return a summary comprising of around 3 sentences.

""".format(article)

    return prompt


def prompt_dsum(dialogue):
    prompt="""
For the following dialogue: {} 

Return a summary comprising of 1 or 2 sentences.

""".format(dialogue)

    return prompt   


def prompt_translation(src):
    prompt = f"Translate the following text from French to English: {src}"
    return prompt



def generate_passkey_prompt(garbage_len=0, passkey_len=100):
    """Generates a text file and inserts an execute line at a random position."""
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_total = garbage_len // 25 
    n_garbage_prefix = random.randint(0, n_garbage_total)
    n_garbage_suffix = n_garbage_total - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there." # 32 tokens
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." # 25 tokens
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = ""
    for _ in range(passkey_len):
        pass_key += random.choice("0123456789") 
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key." # 26 tokens
    final_question = "What is the pass key? The pass key is" # 11 tokens
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def infer_original(input_ids, model, tokenizer, max_steps = 256, max_new_tokens=256, past_key_values=None):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    input_len = input_ids.shape[1]
    outputs = model(input_ids, past_key_values = past_key_values, use_cache=True)
    new_token = 0
    
    for idx in range(max_steps): 
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model(input_id, use_cache=True, past_key_values=outputs.past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break

    return input_ids, new_token, idx, {'16': new_token}, outputs.past_key_values

def infer_quant(input_ids, model, prefill_bit, max_steps=256, past_key_values=None):
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_steps, prefill_bit=prefill_bit, past_key_values=past_key_values)
    return outputs.input_ids, outputs.new_token, outputs.new_token - 1, outputs.precision_log, outputs.past_key_values

def run_eval(
    model_path,
    model_id,
    question_begin,
    question_end,
    answer_file,
    max_new_tokens,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    use_fp,
    benchname,
    high_bit_steps,
    prefill_bit,
    scheduler,
    precisions,
    classifier_path, 
    use_anyprec,
    use_validation=False,
    all_layers=False,
    random_p=None,
):
    if benchname == 'mt_bench':
        questions = load_dataset('HuggingFaceH4/mt_bench_prompts', split='train').to_list()
    elif benchname == 'humaneval':
        questions = read_problems()
        questions = list(questions.values())[question_begin:question_end]
    elif benchname == 'alpaca_eval':
        questions = load_dataset('tatsu-lab/alpaca_eval', split='eval').to_list()
    elif benchname == 'gsm8k':
        if use_validation:
            questions = load_dataset('gsm8k', 'main', streaming=False, split='train')['question'][:100]
        else:
            questions = load_dataset('gsm8k', 'main', streaming=False, split='test')['question']
    elif benchname == 'strategyqa':
        if use_validation:
            questions = load_dataset('ChilleD/StrategyQA', split='train').to_list()
        else:
            questions = load_dataset('ChilleD/StrategyQA', split='test').to_list()
    elif benchname == 'cnn_dm':
        if use_validation:
            dataset = load_dataset("cnn_dailymail", '3.0.0', split='validation')
            questions = dataset['article'][:100]
        else:
            dataset = load_dataset("cnn_dailymail", '3.0.0', split='test')
            questions = dataset['article'][:]
    elif benchname == 'passkey':
        random.seed(0)
        if use_validation:
            iterations = 100
        else:
            iterations = 100
        questions = []
        for i in range(iterations):
            prompt, pass_key = generate_passkey_prompt(garbage_len=200, passkey_len=200)
            questions.append({
            "question_id": i,
            "prompt": prompt,
            "pass_key": pass_key
            })
    elif benchname == 'IWSLT':
        if use_validation:
            dataset = load_dataset('IWSLT/iwslt2017', 'iwslt2017-en-fr', split='validation').to_list()
        else:
            dataset = load_dataset('IWSLT/iwslt2017', 'iwslt2017-en-fr', split='test').to_list()
        questions = []
        tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        for i in range(len(dataset)):
            if len(tokenizer(dataset[i]['translation']['en']).input_ids) > 60:
                questions.append({
                    "id": i,
                    "prompt": prompt_translation(dataset[i]['translation']['fr']),
                    "reference": dataset[i]['translation']['en']
                })
    elif benchname == 'dsum':
        if use_validation:
            dataset = load_dataset('knkarthick/dialogsum', split='validation')
        else:
            dataset = load_dataset('knkarthick/dialogsum', split='test')
        questions = dataset['dialogue']
    else:
        raise ValueError("Unknown benchmark name")

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                use_fp=use_fp,
                benchname=benchname,
                high_bit_steps=high_bit_steps,
                prefill_bit=prefill_bit,
                scheduler=scheduler,
                precisions=precisions,
                classifier_path=classifier_path, 
                use_anyprec=use_anyprec,
                all_layers=all_layers,
                random_p=random_p,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_tokens,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    use_fp,
    benchname,
    high_bit_steps,
    prefill_bit,
    scheduler,
    precisions,
    classifier_path,
    use_anyprec,
    all_layers,
    random_p,
):
    if not use_fp:
        all_precisions = list(set(precisions))
        if prefill_bit is not None and prefill_bit not in precisions:
            all_precisions.append(prefill_bit)
        model = PMPDForCausalLM(model_path, precisions=all_precisions, use_anyprec=use_anyprec).eval().cuda()
        precisions = precisions
        kw_dict = {}
        kw_dict['precisions'] = precisions
        # argument for naive scheduler
        kw_dict['high_bit_steps'] = high_bit_steps
        # argument for kv_cache scheduler
        precision_switch_points = set()
        for p in sorted(precisions, reverse=True):
            if p-1 in precisions:
                precision_switch_points.add((p, p-1))
        kw_dict['precision_switch_points'] = precision_switch_points
        kw_dict['save_dir'] = classifier_path
        if use_anyprec:
            config = model.model.config
        else:
            config = model.model.models[str(precisions[0])].config
        kw_dict['dim'] = config.hidden_size // config.num_attention_heads
        if not all_layers:
            kw_dict['act_dim'] = config.hidden_size  
        else:
            from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
            i = 0
            for m in model.modules():
                if isinstance(m, (torch.nn.Linear, AnyPrecisionLinear)):
                    i +=2 # mean and var
            kw_dict['act_dim'] = i
        kw_dict['num_heads'] = config.num_key_value_heads
        kw_dict['max_new_tokens'] = max_new_tokens
        kw_dict['all_layers'] = all_layers
        kw_dict['random_p'] = random_p
        print('max_new_tokens:', kw_dict['max_new_tokens'])
        # initialize the scheduler
        model.scheduler = Scheduler.get_scheduler(scheduler, **kw_dict)
        tokenizer = model.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)  

    for i, question in tqdm(enumerate(questions), total=len(questions)):
        if benchname == 'mt_bench':
            question_id = question["prompt_id"]
            num_turns = len(question["prompt"])
        elif benchname == 'humaneval':
            question_id = question["task_id"]
            num_turns = 1
        elif benchname == 'alpaca_eval' or benchname == 'gsm8k':
            question_id = i
            num_turns = 1
        elif benchname == 'strategyqa':
            question_id = question["qid"]
            num_turns = 1
        elif benchname == 'cnn_dm':
            question_id = i
            num_turns = 1
        elif benchname == 'passkey':
            question_id = question["question_id"]
            num_turns = 1
        elif benchname == 'IWSLT':
            question_id = question["id"]
            num_turns = 1
        elif benchname == 'dsum':
            question_id = i
            num_turns = 1

        choices = []
        for i in range(num_choices):
            torch.manual_seed(0)
            conv = get_conversation_template('claude-2')
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            precision_logs = []
            for j in range(num_turns):
                past_key_values = None
                if benchname == 'mt_bench':
                    qs = question["prompt"][j]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                elif benchname == 'humaneval':
                    qs = question["prompt"]
                    prompt = qs
                elif benchname == 'alpaca_eval':
                    conv.messages = []
                    conv.append_message(conv.roles[0], question["instruction"])
                    conv.append_message(conv.roles[1], "")
                    prompt = conv.get_prompt()
                elif benchname == 'gsm8k':
                    qs = question
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                elif benchname == 'strategyqa':
                    qs = question["question"]
                    prompt = question['facts'] + '\n' +  qs
                elif benchname == 'cnn_dm':
                    qs = prompt_CNN(question)
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    prompt = 'Play the role of assistant and answer the question from human. ' + prompt
                elif benchname == 'passkey':
                    qs = question["prompt"]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                elif benchname == 'IWSLT':
                    qs = question["prompt"]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    prompt = 'Play the role of assistant and answer the question from human. ' + prompt
                elif benchname == 'dsum':
                    qs = prompt_dsum(question)
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    prompt = 'Play the role of assistant and answer the question from human. ' + prompt
                input_ids = tokenizer([prompt]).input_ids
                input_len = len(input_ids[0])
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    if not use_fp:
                        output_ids, new_token, idx, precision_log, past_key_values = infer_quant(
                            torch.as_tensor(input_ids).cuda(),
                            model,
                            prefill_bit=prefill_bit,
                            max_steps=max_new_tokens,
                            past_key_values=past_key_values
                        )
                    else:
                        output_ids, new_token, idx, precision_log, past_key_values = infer_original(
                            torch.as_tensor(input_ids).cuda(),
                            model,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            past_key_values=past_key_values
                        )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    if benchname == 'mt_bench':
                        
                        # if model.config.is_encoder_decoder:
                        #     output_ids = output_ids[0]
                        # else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                        # be consistent with the template's stop_token_ids
                        if conv.stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in conv.stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]

                        output = tokenizer.decode(
                            output_ids,
                            spaces_between_special_tokens=False,
                        )
                        if conv.stop_str and output.find(conv.stop_str) > 0:
                            output = output[: output.find(conv.stop_str)]
                        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()
                        
                        if conv.name == "xgen" and output.startswith("Assistant:"):
                            output = output.replace("Assistant:", "", 1).strip()
                        conv.messages[-1][-1] = output
                        
                    else:
                        output = tokenizer.decode(
                            output_ids[0].tolist()[input_len:],
                            spaces_between_special_tokens=False,
                        )

                except RuntimeError as e:
                    # print("ERROR question ID: ", question["question_id"])
                    print(e)
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                precision_logs.append(precision_log)
            torch.cuda.empty_cache()        
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time, "precision_log": precision_logs})
        
        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            if benchname == 'passkey':
                ans_json['pass_key'] = question['pass_key']
            elif benchname == 'IWSLT':
                ans_json['reference'] = question['reference']
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--use-fp",
        action='store_true',
        default=False,
        help="Whether to use full precision model. Defaults to False."
    )
    parser.add_argument(
        "--high-bit-steps",
        type=int,
        default=256,
        help="The number of steps to run 8-bit model."
    )
    parser.add_argument(
        "--prefill-bit",
        type=int,
        default=None,
        help="The number of bits for the prefill model."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="naive",
        help="The scheduler to use for precision scheduling."
    )
    parser.add_argument(
        "--precisions",
        type=str,
        default="4,3,2",
        help="The precisions to use for the model."
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="The path to the classifier."
    )
    parser.add_argument(
        "--use-multi-model",
        action='store_true',
        default=False,
        help="Whether to use multiple models."
    )
    parser.add_argument(
        "--validation",
        action='store_true',
        default=False,
        help="Whether to use validation set."
    )
    parser.add_argument(
        "--all_layers",
        action='store_true',
        default=False,
        help="Whether to use all layers for activation scheduler."
    )
    parser.add_argument(
        '--random_p', 
        nargs='+', 
        help='', 
        default=None, 
        required=False)

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    answer_file = args.answer_file
    if args.random_p is not None:
        args.random_p = [float(p) for p in args.random_p]

    print(f"Output to {answer_file}")
    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        use_fp=args.use_fp,
        benchname=args.bench_name,
        high_bit_steps=args.high_bit_steps,
        prefill_bit=args.prefill_bit,
        scheduler=args.scheduler, 
        precisions=[int(p) for p in args.precisions.split(',')],
        classifier_path=args.classifier_path, 
        use_anyprec=not args.use_multi_model,
        use_validation=args.validation,
        all_layers=args.all_layers,
        random_p=args.random_p,
    )

    reorg_answer_file(answer_file)
