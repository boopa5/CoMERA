
gpu=0
use_cuda_graph=0
r=30
emb_tensorized=1
encoder_tensorized=1
n_layers=6
seed=-1


start_epoch=0
epochs_earlystage=20
epochs_latestage=10
target_ratio=0.2

prune_rank=1


batch=32
lr_tensor=1e-4
lr_origin=5e-5
lr_rank=1e-3

print_every=500

# load_path=/network/rit/lab/ziyang_lab/ziyang/github/CoMERA/models_MNLI/MNLI_TensorEnTrue_EmbTrue_NormFalse_lr5e-05_0.0001_acc0.6432.chkpt
model_dir=./models_MNLI/

log_file=./logs/MNLI_batch${batch}_tensorized${encoder_tensorized}_emb${emb_tensorized}_PruneRank${prune_rank}_lr${lr_tensor}_origin${lr_origin}_rank${lr_rank}_cudagraph${use_cuda_graph}.txt


CUDA_VISIBLE_DEVICES=${gpu} python -u MNLI_trainer.py \
      --batch-size ${batch} \
      --start-epoch ${start_epoch} \
      --epochs-earlystage ${epochs_earlystage} \
      --epochs-latestage ${epochs_latestage}\
      --target-ratio ${target_ratio}\
      --model-dir ${model_dir} \
      --lr-tensor ${lr_tensor} \
      --lr-origin ${lr_origin}\
      --lr-rank ${lr_rank} \
      --seed ${seed} \
      --use-cuda-graph ${use_cuda_graph}\
      --max-rank ${r}\
      --n-layers ${n_layers}\
      --emb-tensorized ${emb_tensorized}\
      --encoder-tensorized ${encoder_tensorized}\
      --prune-rank ${prune_rank} \
      --print-every ${print_every}\
      |tee ${log_file}

# --model-loadPATH ${load_path} \

      
