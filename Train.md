
python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --batch_size 1 \
    --crop_size 200 \
    --epochs 400 \
    --warmup_epochs 100 \
    --blr 1.5e-3 \
    --dist_eval \
    --weight_decay 0.05 \


python -m torch.distributed.launch --nproc_per_node=2 only_cnn_main.py     
--batch_size 1     
--crop_size 30     
--epochs 200     
--warmup_epochs 50     
--blr 6e-3     
--dist_eval     
--weight_decay 0.05 \
