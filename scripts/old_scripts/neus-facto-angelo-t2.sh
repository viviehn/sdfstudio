s=c49a8c6cff
s=785e7504b9
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 ns-train neus-facto-angelo \
    --trainer.max-num-iterations 8001  --trainer.steps_per_save 1000\
    --trainer.steps-per-eval-image 2000\
    --pipeline.model.sdf-field.inside-outside True     \
    --pipeline.model.sdf-field.num-layers 2     \
    --pipeline.model.sdf-field.hidden-dim 64     \
    --pipeline.model.sdf-field.geo-feat-dim 64     \
    --pipeline.model.sdf-field.num-layers-color 2     \
    --pipeline.model.sdf-field.log2-hashmap-size 22\
    --pipeline.model.sdf-field.hash-features-per-level 4\
    --pipeline.model.sdf-field.base-res 16\
    --pipeline.model.sdf-field.max-res 2048\
    --pipeline.model.enable-progressive-hash-encoding False\
    --pipeline.model.sdf-field.use-appearance-embedding False\
    --pipeline.model.sdf-field.use-position-encoding True\
    --pipeline.model.sdf-field.vanilla-ngp True\
    --pipeline.model.sdf-field.geometric-init False\
    --pipeline.model.sdf-field.bias 0.8\
    --pipeline.model.sdf-field.fix-geonet False\
    --pipeline.model.sdf-field.use-numerical-gradients False\
    --pipeline.model.background-model none\
    --pipeline.model.sparse_points_sdf_loss_mult 1.0\
    --pipeline.model.curvature-loss-warmup-steps 2000\
    --pipeline.model.curvature-loss-multi 0.0\
    --pipeline.model.eikonal-loss-mult 0.0\
    --pipeline.datamanager.train_num_rays_per_batch 2\
    --pipeline.datamanager.train_num_images_to_sample_from -1\
    --pipeline.datamanager.train_num_times_to_repeat_images -1\
    --pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    --experiment-name ngpc-best-2optim-40m-f4-l1-rgb.1     sdfstudio-data \
    --data /home/fangyin/data/scannetpp/$s/dslr/sdfstudio\
    --include_sdf_samples True\
    --use_point_color True
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 ns-train neus-facto-angelo \
    #--trainer.max-num-iterations 6101  --trainer.steps_per_save 1000\
    #--trainer.steps-per-eval-image 2000\
    #--pipeline.model.sdf-field.inside-outside True     \
    #--pipeline.model.sdf-field.num-layers 2     \
    #--pipeline.model.sdf-field.hidden-dim 64     \
    #--pipeline.model.sdf-field.geo-feat-dim 64     \
    #--pipeline.model.sdf-field.num-layers-color 2     \
    #--pipeline.model.sdf-field.log2-hashmap-size 22\
    #--pipeline.model.sdf-field.hash-features-per-level 4\
    #--pipeline.model.sdf-field.base-res 16\
    #--pipeline.model.sdf-field.max-res 2048\
    #--pipeline.model.enable-progressive-hash-encoding False\
    #--pipeline.model.sdf-field.use-appearance-embedding True\
    #--pipeline.model.sdf-field.use-position-encoding True\
    #--pipeline.model.sdf-field.vanilla-ngp True\
    #--pipeline.model.sdf-field.geometric-init False\
    #--pipeline.model.sdf-field.bias 0.8\
    #--pipeline.model.sdf-field.fix-geonet False\
    #--pipeline.model.sdf-field.use-numerical-gradients True\
    #--pipeline.model.background-model none\
    #--pipeline.model.sparse_points_sdf_loss_mult 1.0\
    #--pipeline.model.curvature-loss-warmup-steps 2000\
    #--pipeline.model.curvature-loss-multi 0.0001\
    #--pipeline.datamanager.train_num_rays_per_batch 2\
    #--pipeline.datamanager.train_num_images_to_sample_from -1\
    #--pipeline.datamanager.train_num_times_to_repeat_images -1\
    #--pipeline.datamanager.eval_num_images_to_sample_from 1 --vis tensorboard\
    #--experiment-name ngp-best-nema-500m-f4-l1-2loss-cur2000     sdfstudio-data \
    #--data /home/fangyin/data/scannetpp/785e7504b9/dslr/sdfstudio\
    #--include_sdf_samples True
    #--pipeline.model.curvature-loss-multi 0.0\
    #--pipeline.model.eikonal-loss-mult 0.0\
    #--pipeline.model.steps-per-level 800\
    #--pipeline.model.sdf-field.geometric-init True\
    #--trainer.load-dir outputs/ngp-best/neus-facto-angelo/2024-02-26_105859/sdfstudio_models \
    #--pipeline.model.sdf-field.use-numerical-gradients True\
    #--trainer.steps-per-eval-batch 2\
