export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=./:$PYTHONPATH

# python src/evaluate/iqa_eval_vqa.py \
# 	--model-path MAGAer13/mplug-owl2-llama2-7b \
# 	--save-dir results/mplug/ \
# 	--preprocessor-path ./preprocessor/ \
# 	--root-dir ./data/ \
# 	--meta-paths ./data/GenAI-Bench/metas/all_alignment.json

# python src/evaluate/iqa_eval_vqa_current.py \
#   --model-path ./checkpoints/deqa_lora \
#   --model-base zhiyuanyou/DeQA-Score-Mix3 \
#   --save-dir results/deqa_lora/ \
#   --preprocessor-path ./preprocessor/ \
#   --meta-paths ./data/GenAI-Bench/metas/all_alignment.json

python src/evaluate/iqa_eval_vqa_current.py \
  --model-path ./checkpoints/test \
  --model-base zhiyuanyou/DeQA-Score-Mix3 \
  --save-dir results/test/ \
  --preprocessor-path ./preprocessor/ \
  --meta-paths ./data/AIGCIQA2023/metas/val_alignment.json

	# ../data/KONIQ/metas/test_koniq_2k.json \
				#  ../data/SPAQ/metas/test_spaq_2k.json \
				#  ../data/KADID10K/metas/test_kadid_2k.json \
				#  ../data/PIPAL/metas/test_pipal_5k.json \
				#  ../data/LIVE-WILD/metas/test_livew_1k.json \
				#  ../data/AGIQA3K/metas/test_agiqa_3k.json \
				#  ../data/TID2013/metas/test_tid2013_3k.json \
				#  ../data/CSIQ/metas/test_csiq_866.json \

#MAGAer13/mplug-owl2-llama2-7b 
# --level-names excellent good fair poor bad \
