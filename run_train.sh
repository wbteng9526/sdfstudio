ns-train nerfacto \
    --timestamp 1 \
    --trainer.max-num-iterations 5000 \
    --viewer.quit-on-train-completion True \
    sdfstudio-data \
    --data ./data/dtu/scan63/

ns-export tsdf \
    --load-config outputs/data-dtu-scan63/nerfacto/1/config.yml \
    --output-dir outputs/data-dtu-scan63/nerfacto/1/

ns-export poisson \
    --load-config outputs/data-dtu-scan63/nerfacto/1/config.yml \
    --output-dir outputs/data-dtu-scan63/nerfacto/1/