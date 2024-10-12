import json
# argparse
import sys
from datasets import load_dataset
import re
import evaluate
from collections import defaultdict


def parse(text):
  # return the parts of the text after ASSISTANT: 
  parts = text.split("ASSISTANT: ")
  if len(parts) == 1:
    parts = text.split("ASSISTANT:")
  if len(parts) == 1:
    parts = text.split("Assistant:")
  if len(parts) == 1:
    parts = text.split("[/INST]")
  # remove </s>
  answer = ' '.join(parts[-1].split()).replace("</s>", "")
  return answer


def get_integer_string(answer):
  # remove non-numeric characters
  answer = ''.join(filter(str.isdigit, answer))
  return answer

# Find the length of common prefix of two strings
def common_prefix_length(str1, str2):
  i = 0
  while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
    i += 1
  return i


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
  if bench_name == "mt_bench" or bench_name == "cnn_dm":
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")
  bert_scores = []
  bert_precisions = []
  rouge_scores = []
  rouge_precisions = []
  with open(input_files[0], "r") as f:
    data = [json.loads(line) for line in f]
  for i, _ in enumerate(data):
    best_bert_score = None
    best_rouge_score = None
    best_bert_precision = None
    best_rouge_precision = None
    for input_file in input_files:
        with open(input_file, "r") as f:
            d = [json.loads(line) for line in f][i]
        prec_dict = defaultdict(int)
        if bench_name == "mt_bench" or bench_name == "cnn_dm":
          summaries = []
          references = []
        elif bench_name == "passkey" or bench_name == "gsm8k":
          correct = 0
        for j in range(len(d["choices"][0]["turns"])):
          answer = parse(d["choices"][0]["turns"][j])
          if bench_name == "cnn_dm":
            summaries.append(answer)
            references.append(reference_data[i])
          if bench_name == "mt_bench":
            summaries.append(answer)
            references.append(reference_data[i]["choices"][0]["turns"][j])
          elif bench_name == "passkey":
            answer = get_integer_string(answer)
            correct += (common_prefix_length(answer, d["pass_key"]) / len(d["pass_key"]))
          elif bench_name == "gsm8k":
            n = extract_integer_string(parse(answer))
            reference_answer = extract_integer_string(reference_data[i])
            if n is not None and (n == reference_answer or abs(float(n) - int(reference_answer)) < 1e-6):
              correct += 1
          for prec, step in d['choices'][0]['precision_log'][j].items():
            prec_dict[prec] += step
          # weighted average of precision
        prec = sum([int(p) * step for p, step in prec_dict.items()]) / sum(prec_dict.values())
        if bench_name == "mt_bench" or bench_name == "cnn_dm":
          rouge_score = rouge.compute(predictions=summaries, references=references, use_stemmer=True)['rougeL']
          bert_score = bert.compute(predictions=summaries, references=references, lang="en")
          bert_score = sum(bert_score['f1']) / len(bert_score['f1'])
          score = bert_score
        elif bench_name == "passkey" or bench_name == "gsm8k":
          score = correct
        # if best_score is None or best_score < score:
        #   best_score = score
        #   best_precision = prec
        if best_rouge_score is None or best_rouge_score < rouge_score:
          best_rouge_score = rouge_score
          best_rouge_precision = prec
        if best_bert_score is None or best_bert_score < bert_score:
          best_bert_score = bert_score
          best_bert_precision = prec
    bert_scores.append(best_bert_score)
    bert_precisions.append(best_bert_precision)
    rouge_scores.append(best_rouge_score)
    rouge_precisions.append(best_rouge_precision)
  return sum(bert_scores) / len(bert_scores), sum(bert_precisions) / len(bert_precisions), sum(rouge_scores) / len(rouge_scores), sum(rouge_precisions) / len(rouge_precisions)


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
  else:
    raise ValueError("Bench name not recognized")
  validation = False
  for f in input_files:
    if 'validation' in f:
      validation = True
      break
  
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

  input_files = [f for f in input_files if "fp16" not in f]
  
  bert_score, bert_precision, rouge_score, rouge_precision = get_scores(input_files, bench_name, reference_data)
  print("bert_score =", bert_score)
  print("bert_precision =", bert_precision)
  print("rouge_score =", rouge_score)
  print("rouge_precision =", rouge_precision)
  
  