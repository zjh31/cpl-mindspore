

### Installation
2.  Prepare for the running environment. 

    You can either use the docker image we provide, or follow the installation steps in [`ReSC`](https://github.com/zyang-ur/ReSC). 

    ```
    docker pull djiajun1206/vg:pytorch1.5
    ```

### Getting Started

Please refer to [GETTING_STARGTED.md](docs/GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints.

### Training and Evaluation

1.  Training
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50.ckpt --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50 --epochs 20
    ``` 

2.  Evaluation
    ```
    CUDA_VISIBLE_DEVICES=0 python eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/unc_r50/best_checkpoint.ckpt --output_dir ./outputs/unct_r50
    ``
