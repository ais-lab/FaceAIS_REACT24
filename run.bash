# Description: This file contains the commands to run the experiments in the paper

# PLEASE CHANGE EVERY PATH TO YOUR OWN PATH

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# train and test the face tokenizer, can add --resume and --resume_ckpt to resume training

# python p1_face_tokenizer.py --run_name "quantize_vq_pretrain_3-lfqtmu" --quantize_type "lfq" --output_dir "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3" --dataset_path "/home/tien/playground_facereconstruction/data/react_2024"

# # train predictor, should change the parameter inside the main function of p2_react_predictor.py
# # can add --resume and --resume_checkpoint to resume training

# python p2_react_predictor.py --output_dir /home/tien/playground_facereconstruction/output/ --tokenizer_checkpoint /home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=149-step=30000-v1.ckpt --dataset_path /home/tien/playground_facereconstruction/data/react_2024

# # test predictor, this should take very long as it needs to render the video.
# # in the end of the file, there are a few lines to test the predictor, you can comment the part that you don't want to run

# python p2_react_predictor.py --test --test_checkpoint /home/tien/playground_facereconstruction/output/epoch=17-step=1800.ckpt --tokenizer_checkpoint /home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=155-step=31200.ckpt --output_dir /home/tien/playground_facereconstruction/output/ --dataset_path /home/tien/playground_facereconstruction/data/react_2024

# if there is prediction folder ready, you can run the following command to compute the metrics

# python compute_metrics.py --output_dir /home/tien/playground_facereconstruction/output/lfq-wav2vec2-patch32-context256-vocab2048/k512-1-p32-output/test/render_for_fid/ --dataset_path /home/tien/playground_facereconstruction/data/react_2024 --val_test test

# python compute_metrics.py --output_dir /home/tien/playground_facereconstruction/output/lfq-wav2vec2-patch32-context256-vocab2048/k512-1-p32-output/val/render_for_fid/ --dataset_path /home/tien/playground_facereconstruction/data/react_2024 --val_test val