import json
# argparse
import sys
from datasets import load_dataset
import re
import evaluate
from collections import defaultdict


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
      for i, d in enumerate(data):
        prec_dict = defaultdict(int)
        for j in range(len(d["choices"][0]["turns"])):
          answer = d["choices"][0]["turns"][j]
          if bench_name == "cnn_dm":
            summaries.append(answer)
            references.append(reference_data[i])
          elif bench_name == "mt_bench":
            summaries.append(answer)
            references.append(reference_data[i]["choices"][0]["turns"][j])
          elif bench_name == "dsum":
            summaries.append(answer)
            references.append(reference_data[i])
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
  elif bench_name == "dsum":
    if validation:
      reference_data = load_dataset('knkarthick/dialogsum', split='validation')['summary']
    else:
      reference_data = load_dataset('knkarthick/dialogsum', split='test')['summary']

  scores, len_dict, precision_dict = get_scores(input_files, bench_name, reference_data)
  print("score_dict =", scores)
  print("len_dict =", len_dict)
  print("precision_dict =", precision_dict)
  
  