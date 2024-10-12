import json
# argparse
import sys
from datasets import load_dataset
import re
import evaluate
from collections import defaultdict


def parse(text, part_index=-1):
  # return the parts of the text after ASSISTANT: 
  parts = text.split("ASSISTANT: ")
  if len(parts) == 1:
    parts = text.split("ASSISTANT:")
  if len(parts) == 1:
    parts = text.split("Assistant:")
  if len(parts) == 1:
    parts = text.split("[/INST]")
  # remove </s>
  answer = ' '.join(parts[part_index].split()).replace("</s>", "")
  return answer


def get_integer_string(answer):
  # find the first continuous integer chunk in the string
  i = 0
  while i < len(answer) and not answer[i].isdigit():
    i += 1
  j = i
  while j < len(answer) and answer[j].isdigit():
    j += 1
  answer = answer[i:j]
  # print(len(answer), answer)
  return answer

# Find the length of common prefix of two strings
def common_prefix_length(str1, str2):
  i = 0
  while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
    i += 1
  return i


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def extract_integer_string(answer):
    # remove ,
    answer = answer.replace(',', '')
    # Find all numeric substrings in the answer, including floating-point numbers
    numeric_strings = re.findall(r'\-?\d+\.\d+|\-?\d+', answer)
    if numeric_strings:
        # Get the last numeric string and convert it to float
        return float(numeric_strings[-1])
    else:
        # Return None or raise an error if no numeric string is found
        return None


def get_scores(input_files, bench_name, reference_data=None):
  if bench_name == "mt_bench" or bench_name == "cnn_dm" or bench_name == "dsum":
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")
  elif bench_name == 'IWSLT':
    bleu =  evaluate.load("bleu")
    bleurt = evaluate.load("bleurt")
    sacre_bleu = evaluate.load("sacrebleu")
  score_dict = {}
  len_dict = {}
  precision_dict = {}
  for input_file in input_files:
      with open(input_file, "r") as f:
          data = [json.loads(line) for line in f]
          print("len(data) =", len(data))
      new_tokens = []
      precisions = []
      if bench_name == "mt_bench" or bench_name == "cnn_dm" or bench_name == "IWSLT" or bench_name == "dsum":
        summaries = []
        references = []
      elif bench_name == "passkey" or bench_name == "gsm8k":
        correct = 0
      for i, d in enumerate(data):
        prec_dict = defaultdict(int)
        for j in range(len(d["choices"][0]["turns"])):
          answer = d["choices"][0]["turns"][j]
          if bench_name == "cnn_dm":
            answer = parse(answer, 1)
            summaries.append(answer)
            references.append(reference_data[i])
          elif bench_name == "mt_bench":
            summaries.append(answer)
            references.append(reference_data[i]["choices"][0]["turns"][j])
          elif bench_name == "dsum":
            summaries.append(answer)
            references.append(reference_data[i])
          elif bench_name == "passkey":
            answer = parse(answer, 1)
            answer = get_integer_string(answer)
            # print(common_prefix_length(answer, d["pass_key"]))
            # print(answer)
            correct += answer == d["pass_key"]
          elif bench_name == "gsm8k":
            # n = extract_integer_string(parse(answer, 1))
            # reference_answer = extract_integer_string(reference_data[i])
            # if n is not None and (n == reference_answer or abs(float(n) - int(reference_answer)) < 1e-6):
            #   correct += 1
            n = extract_answer_from_output(parse(answer, 1))
            reference_answer = extract_answer_from_output(reference_data[i])
            correct += is_correct(n, reference_answer)
          elif bench_name == "IWSLT":
            summaries.append(answer)
            references.append(d["reference"])
          for prec, step in d['choices'][0]['precision_log'][j].items():
            prec_dict[prec] += step
          # weighted average of precision
        if prec_dict:
          precisions.append(sum([int(p) * step for p, step in prec_dict.items()]) / sum(prec_dict.values()))
        # precisions.append((sum([int(p) * step for p, step in prec_dict.items()]), sum(prec_dict.values())))
            
        new_tokens.extend(d["choices"][0]["new_tokens"])
      if bench_name == "mt_bench" or bench_name == "cnn_dm" or bench_name == "dsum":
        rouge_score = rouge.compute(predictions=summaries, references=references, use_stemmer=True)
        # aggregate bert score
        bert_score = bert.compute(predictions=summaries, references=references, lang="en")
        # merge the scores
        score_dict[input_file] = {
          "rougeL": rouge_score["rougeL"],
          "bert_score": sum(bert_score['f1']) / len(bert_score['f1'])
        }
      elif bench_name == "passkey" or bench_name == "gsm8k":
        score_dict[input_file] = {
          "accuracy": correct / len(data),
        }
      elif bench_name == "IWSLT":
        bleu_score = bleu.compute(predictions=summaries, references=references)
        bleurt_score = bleurt.compute(predictions=summaries, references=references)
        sacre_bleu_score = sacre_bleu.compute(predictions=summaries, references=references)
        score_dict[input_file] = {
          "bleu": bleu_score['bleu'],
          "bleurt": sum(bleurt_score['scores']) / len(bleurt_score['scores']),
          "sacre_bleu": sacre_bleu_score['score']
        }
      len_dict[input_file] = sum(new_tokens) / len(new_tokens)
      # precision_dict[input_file] = sum([p[0] for p in precisions]) / sum([p[1] for p in precisions])
      precision_dict[input_file] = sum(precisions) / len(precisions)
  
  return score_dict, len_dict, precision_dict


if __name__ == '__main__':
  input_files = sys.argv[1:]
  bench_name = None 
  if 'mt_bench' in input_files[0]:
    bench_name = "mt_bench"
  elif 'cnn_dm' in input_files[0]:
    bench_name = "cnn_dm"
  elif 'passkey' in input_files[0]:
    bench_name = "passkey"
  elif 'gsm8k' in input_files[0]:
    bench_name = "gsm8k"
  elif 'IWSLT' in input_files[0]:
    bench_name = "IWSLT"
  elif 'dsum' in input_files[0]:
    bench_name = "dsum"
  else:
    raise ValueError("Bench name not recognized")
  validation = False
  for f in input_files:
    if 'validation' in f:
      validation = True
      break
  # validation = False
  print("validation =", validation)
  
  reference_data = None
  if bench_name == "mt_bench":
    # reference file is the one with fp16 in its name
    reference_file = [f for f in input_files if "fp16" in f][0]
    with open(reference_file, "r") as f:
      reference_data = [json.loads(line) for line in f]
  elif bench_name == "cnn_dm":
    if validation:
      reference_data = load_dataset("cnn_dailymail", "3.0.0")["validation"]["highlights"]
    else:
      reference_data = load_dataset("cnn_dailymail", "3.0.0")["test"]["highlights"]
  elif bench_name == "gsm8k":
    if validation:
      reference_data = load_dataset("gsm8k", 'main')['train']['answer']
    else:
      reference_data = load_dataset("gsm8k", 'main')['test']['answer']
  elif bench_name == "dsum":
    if validation:
      reference_data = load_dataset('knkarthick/dialogsum', split='validation')['summary']
    else:
      reference_data = load_dataset('knkarthick/dialogsum', split='test')['summary']

  scores, len_dict, precision_dict = get_scores(input_files, bench_name, reference_data)
  print("score_dict =", scores)
  print("len_dict =", len_dict)
  print("precision_dict =", precision_dict)
  
  