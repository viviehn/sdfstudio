s=0a7cc12c0e
s=0e75f3c4d9
s=c49a8c6cff
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 ns-train neus-facto  \
    --trainer.max_num_iterations 200000 \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.8  \
    --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.mono-depth-loss-mult 0.1 \
    --pipeline.model.mono-normal-loss-mult 0.05 \
    --pipeline.datamanager.train_num_rays_per_batch 2048 \
    --pipeline.datamanager.eval_num_rays_per_batch 1024 \
    --pipeline.datamanager.train_num_images_to_sample_from 128 \
    --pipeline.datamanager.train_num_times_to_repeat_images 100 \
    --pipeline.datamanager.eval_num_images_to_sample_from 16 \
    --vis tensorboard --experiment-name neus-facto-mono256_100-$s \
    sdfstudio-data \
    --data /data/fwei/scannetpp/data/$s/dslr/sdfstudio \
    --include_mono_prior True
    #--trainer.load-dir outputs/neus-facto-mono256_100/neus-facto/2023-12-06_034653/sdfstudio_models\
