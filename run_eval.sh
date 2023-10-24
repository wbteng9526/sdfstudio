ns-render-mesh-eval \
    --meshfile outputs/neus-facto-dtu65/neus-facto/2023-10-19_150745/tsdf_mesh.ply \
    --load_config outputs/neus-facto-dtu65/neus-facto/2023-10-19_150745/config.yml \
    --data_dir /home/wteng/workspace/csci677/data/DTU/scan65/ \
    --camera_path_filename meta_data.json \
    --output_dir outputs/neus-facto-dtu65/neus-facto/2023-10-19_150745/eval \
    --eval_output_path eval.json \
    --traj filename