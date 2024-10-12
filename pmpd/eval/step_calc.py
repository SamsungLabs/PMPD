import json
# argparse
import sys
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
  assert len(parts) >= 2, f"Could not split text: {text}"
  # remove </s>
  answer = ' '.join(parts[-1].split()).replace("</s>", "")
  return answer


def get_len(input_files):
    step_dict = {}
    precision_dict = {}
    len_dict = {}
    for input_file in input_files:
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
        new_tokens = []
        precisions = []
        steps = defaultdict(float)
        for i, d in enumerate(data):
          new_tokens.extend(d["choices"][0]["new_tokens"])
          prec_dict = {}
          for i in range(len(d['choices'][0]['precision_log'])):
            for prec, step in d['choices'][0]['precision_log'][i].items():
              prec_dict[prec] = step
              if steps[prec] == 0:
                print(prec)
              steps[prec] += step
            # weighted average of precision
          if len(prec_dict) > 0:
            precisions.append(sum([int(p) * step for p, step in prec_dict.items()]) / sum(prec_dict.values()))
          else: 
            # precisions.append(3)
            pass
          # precisions.append((sum([int(p) * step for p, step in prec_dict.items()]), sum(prec_dict.values())))
        print(steps)
        step_dict[input_file] = {prec : round(steps / len(new_tokens),1) for prec, steps in steps.items()}
        precision_dict[input_file] = sum(precisions) / len(precisions)
        len_dict[input_file] = sum(new_tokens) / len(new_tokens)
    # sort step_dict according to key values
    # step_dict = dict(sorted(step_dict.items(), key=lambda item: item[0]))
    return step_dict, precision_dict, len_dict

if __name__ == '__main__':
  input_files = sys.argv[1:]
  
  step_dict, precision_dict, len_dict = get_len(input_files)
  print('Bit steps', step_dict)
  print('Precision', precision_dict)
  print('Total tokens', len_dict)