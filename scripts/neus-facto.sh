s=785e7504b9
s=0a76e06478
s=c49a8c6cff
s=0a7cc12c0e
s=0e75f3c4d9
CUDA_VISIBLE_DEVICES=1 ns-train neus-facto  --trainer.max_num_iterations 200000 \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.8  --pipeline.model.sdf-field.inside-outside True \
    --pipeline.datamanager.train_num_rays_per_batch 2048 --vis tensorboard \
    --experiment-name neus-facto-$s sdfstudio-data \
    --data /data/fwei/scannetpp/data/$s/dslr/sdfstudio 
    #--viewer.websocket-port 16008 
