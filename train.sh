RETRAIN_NUM=6
GPU_NUM=2
dataset='unc'


for((i=1;i<=${RETRAIN_NUM};i++));do
if ((i==0));then
# train
python -m torch.distributed.launch   --nproc_per_node=${GPU_NUM}  --use_env train.py  --batch_size 28 --dataset=${dataset} \
--output_dir outputs/${dataset}/


else
python -m torch.distributed.launch   --nproc_per_node=${GPU_NUM}  --use_env train.py  --batch_size 28 --dataset=${dataset} \
--output_dir ./outputs/${dataset}/ --resume ./outputs/${dataset}/checkpoint0019.pth

fi
done
