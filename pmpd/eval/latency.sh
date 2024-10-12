export CUDA_VISIBLE_DEVICES=1

echo MobileLLaMA-1.4B-Chat-8-2
echo 8
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 8
echo 7
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 7
echo 6
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 6
echo 5
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 5
echo 4
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 4
echo 3
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 3
echo 2
python pmpd/eval/latency.py --model-path anyprec-MobileLLaMA-1.4B-Chat-8-2/ --precision 2