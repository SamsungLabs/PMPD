import subprocess
import argparse
from multiprocessing import Pool

# Parameters
GPUS = [0, 1, 2, 3]
STEPS = [0, 60, 120, 160]
MODEL_ID = "MobileLLaMA"
MODEL_PATH = "anyprec-MobileLLaMA-1.4B-Chat-4-2"
BENCH_NAME = "cnn_dm"
QUESTION_END = 1000
PRECISION_HIGH = 4
PRECISION_LOW = 3
PRECISIONS = f"{PRECISION_HIGH},{PRECISION_LOW}"
ANSWER_FILE_DIR = f"data/anyprec/{BENCH_NAME}/"
CLASSIFIER_PATH = f"test/{MODEL_PATH}_ts1_tk3_2048"
FP_MODEL_PATH = "anyprec-MobileLLaMA-1.4B-Chat-4-2"
HIGH_BIT_STEPS = 0
MAX_NEW_TOKENS = 256

# Test control
SEARCH = 0
STATIC_SEARCH = 0
RANDOM_UNIFORM = 0
RANDOM_PRIOR = 0
PRIOR_DISTRIBUTION = 0
BASELINE = 0
FP16 = 0
KV_SCHEDULER = 0
CONFIDENCE_SCHEDULER = 0
STATIC_SCHEDULER = 0 


def run_search(step, gpu):
    """Run the evaluation job for the specified step and GPU."""
    command = [
        "CUDA_VISIBLE_DEVICES={}".format(gpu),
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{PRECISION_HIGH}-{PRECISION_LOW}/{MODEL_ID}-schedule-{step}.jsonl",
        "--high-bit-steps", str(step),
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)


def run_static_search(step, gpu):
    """Run the evaluation job for the specified step and GPU."""
    command = [
        "CUDA_VISIBLE_DEVICES={}".format(gpu),
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{PRECISION_HIGH}-{PRECISION_LOW}/{MODEL_ID}-schedule-{step}-validation.jsonl",
        "--high-bit-steps", str(step),
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--validation"
    ]
    subprocess.run(" ".join(command), shell=True, check=True)


def run_random_uniform():
    """Run the Random Uniform scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-random_uniform.jsonl",
        "--scheduler", "random",
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)


def run_random_prior():
    """Run the Random Prior scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=1",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-random_prior.jsonl",
        "--scheduler", "random",
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--random_p", " ".join(PRIOR_DISTRIBUTION)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)


def run_baseline_jobs():
    """Run individual precision jobs."""
    commands = [
        [
            "CUDA_VISIBLE_DEVICES=0",
            "python3", "pmpd/eval/evaluate_generation.py",
            "--model-path", MODEL_PATH,
            "--model-id", MODEL_ID,
            "--bench-name", BENCH_NAME,
            "--question-end", str(QUESTION_END),
            "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-{PRECISION_HIGH}bit.jsonl",
            "--high-bit-steps", "0",
            "--precisions", str(PRECISION_HIGH),
            "--max-new-tokens", str(MAX_NEW_TOKENS)
        ],
        [
            "CUDA_VISIBLE_DEVICES=1",
            "python3", "pmpd/eval/evaluate_generation.py",
            "--model-path", MODEL_PATH,
            "--model-id", MODEL_ID,
            "--bench-name", BENCH_NAME,
            "--question-end", str(QUESTION_END),
            "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-{PRECISION_LOW}bit.jsonl",
            "--high-bit-steps", "0",
            "--precisions", str(PRECISION_LOW),
            "--max-new-tokens", str(MAX_NEW_TOKENS)
        ]
    ]
    for command in commands:
        subprocess.run(" ".join(command), shell=True, check=True)

def run_fp16_jobs():
    """Run the FP16 scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", FP_MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-fp16.jsonl",
        "--use-fp", 
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)

def run_kv_scheduler():
    """Run the KV Cache scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=2",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-kv_cache-{PRECISION_HIGH}-{PRECISION_LOW}.jsonl",
        "--scheduler", "kv_cache",
        "--precisions", PRECISIONS,
        "--classifier_path", CLASSIFIER_PATH,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)

def run_confidence_scheduler():
    """Run the Confidence scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=2",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-confidence-{PRECISION_HIGH}-{PRECISION_LOW}.jsonl",
        "--scheduler", "confidence",
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)

def distribute_job(run_job):
    """Distribute jobs across GPUs."""
    with Pool(processes=len(GPUS)) as pool:
      for i, step in enumerate(STEPS):
          gpu_index = i % len(GPUS)
          pool.apply_async(run_job, (step, GPUS[gpu_index]))
      pool.close()
      pool.join()

def run_static_scheduler():
    """Run the static scheduler."""
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3", "pmpd/eval/evaluate_generation.py",
        "--model-path", MODEL_PATH,
        "--model-id", MODEL_ID,
        "--bench-name", BENCH_NAME,
        "--question-end", str(QUESTION_END),
        "--answer-file", f"{ANSWER_FILE_DIR}/{MODEL_ID}-static-{PRECISION_HIGH}-{PRECISION_LOW}.jsonl",
        "--high-bit-steps", str(HIGH_BIT_STEPS),
        "--precisions", PRECISIONS,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    subprocess.run(" ".join(command), shell=True, check=True)

def main():
    if SEARCH:
        distribute_job(run_search)
    
    if STATIC_SEARCH:
        distribute_job(run_static_search)
    
    if STATIC_SCHEDULER:
        run_static_scheduler()
        
    if RANDOM_UNIFORM:
        run_random_uniform()
    
    if RANDOM_PRIOR:
        run_random_prior()
      
    if BASELINE:
        run_baseline_jobs()

    if FP16:
        run_fp16_jobs()

    if KV_SCHEDULER:
        run_kv_scheduler()

    if CONFIDENCE_SCHEDULER:
        run_confidence_scheduler()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--static-search", action="store_true", default=False)
    argparser.add_argument("--search", action="store_true", default=False)
    argparser.add_argument("--static-scheduler", action="store_true", default=False)
    argparser.add_argument("--random-uniform", action="store_true", default=False)
    argparser.add_argument("--random-prior", action="store_true", default=False)
    argparser.add_argument("--prior-distribution", nargs='+', default=None, required=False)
    argparser.add_argument("--baseline", action="store_true", default=False)
    argparser.add_argument("--fp16", action="store_true", default=False)
    argparser.add_argument("--kv-scheduler", action="store_true", default=False)
    argparser.add_argument("--confidence-scheduler", action="store_true", default=False)
    argparser.add_argument("--model-id", type=str, default="MobileLLaMA")
    argparser.add_argument("--model-path", type=str, default="anyprec-MobileLLaMA-1.4B-Chat-4-2")
    argparser.add_argument("--bench-name", type=str, default="cnn_dm")
    argparser.add_argument("--question-end", type=int, default=1000)
    argparser.add_argument("--precision-high", type=int, default=4)
    argparser.add_argument("--precision-low", type=int, default=3)
    argparser.add_argument("--steps", type=str, default="0,60,120,160")
    argparser.add_argument("--gpus", type=str, default="0,1,2,3")
    argparser.add_argument("--answer-file-dir", type=str, default="data/anyprec/")
    argparser.add_argument("--classifier-path", type=str, default="test/anyprec-MobileLLaMA-1.4B-Chat-4-2_ts1_tk3_2048")
    argparser.add_argument("--fp-model-path", type=str, default="mtgv/MobileLLaMA-1.4B-Chat")
    argparser.add_argument("--high-bit-steps", type=int, default=0)
    argparser.add_argument("--max-new-tokens", type=int, default=256)
    args = argparser.parse_args()
    
    SEARCH = args.search
    STATIC_SEARCH = args.static_search
    STATIC_SCHEDULER = args.static_scheduler
    BASELINE = args.baseline
    FP16 = args.fp16
    KV_SCHEDULER = args.kv_scheduler
    CONFIDENCE_SCHEDULER = args.confidence_scheduler
    RANDOM_PRIOR = args.random_prior
    RANDOM_UNIFORM = args.random_uniform
    
    MODEL_ID = args.model_id
    MODEL_PATH = args.model_path
    BENCH_NAME = args.bench_name
    QUESTION_END = args.question_end
    PRECISION_HIGH = args.precision_high
    PRECISION_LOW = args.precision_low
    PRECISIONS = f"{PRECISION_HIGH},{PRECISION_LOW}"
    ANSWER_FILE_DIR = f'{args.answer_file_dir}/{BENCH_NAME}/'
    CLASSIFIER_PATH = args.classifier_path
    PRIOR_DISTRIBUTION = args.prior_distribution
    FP_MODEL_PATH = args.fp_model_path
    HIGH_BIT_STEPS = args.high_bit_steps
    MAX_NEW_TOKENS = args.max_new_tokens
    
    STEPS = [int(step) for step in args.steps.split(",")]
    GPUS = [int(gpu) for gpu in args.gpus.split(",")]

    main()
