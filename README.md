This is the submission for the REACT2024 competition from [AIS lab, Ritsumeikan University, Japan](http://www.aislab.org/). 

### Content List

- [Team Members](#team-members)
- [Introduction](#introduction)
- [Results](#results)
  - [Output comparison with Learning2Listen's baseline:](#output-comparison-with-learning2listens-baseline)
- [Prequisites](#prequisites)
- [Installation](#installation)
- [Download checkpoints, external and dataset](#download-checkpoints-external-and-dataset)
- [Model input and output](#model-input-and-output)
- [Dataloader](#dataloader)
- [Training](#training)
  - [Command](#command)
- [Metrics harvesting](#metrics-harvesting)
  - [Command](#command-1)
- [Visualization](#visualization)
- [Data](#data)
- [Known issues](#known-issues)
- [Acknowledgements](#acknowledgements)

## Team Members
- Dam Quang Tien
- Nguyen Tri Tung Nguyen
- Tran Dinh Tuan
- Lee Joo-Ho

## Introduction

We propose a novel approach to solve the problem of the REACT2024 competition in **Online** Appropriate Facial Reaction Generation. Our approach is based on the combination of the following techniques:
1. Finite Scalar Quantization ([FSQ](https://github.com/google-research/google-research/tree/master/fsq)) allows better codebook usability, making a better face tokenization method and enabling us to create a single-frame face tokenization method.
2. Cross-modality attention ([FIBER](https://github.com/microsoft/FIBER)) allows better feature fusion between the speaker's voice and facial expression, and fusion between the speaker's feature and the listener's past feature. This allows it to generate consistent but dynamic reactions.

## Results

The model's performance on the validation set is as follows: 
|                              | Appropriateness |        | Diversity |        |        | Realism | Synchrony |
|------------------------------|:---------------:|:------:|:---------:|:------:|:------:|:-------:|:---------:|
|                              | FRC             | FRD    | FRDvs     | FRVar  | FRDiv  | FRRea   | FRSyn     |
| Ground truth                 | 8.35            | 0      | 0.2117    | 0.0635 | 0      | -       | 48.42     |
| Mime ~ speaker lagged        | 0.37            | 84.78  | 0.2117    | 0.0635 | 0.2483 | -       | 43.50     |
| Random                       | 0.05            | 228.33 | 0.1666    | 0.0833 | 0.1666 | -       | 46.27     |
| Trans-VAE online             | 0.07            | 90.31  | 0.0064    | 0.0012 | 0.0009 | 69.19   | 44.65     |
| BeLFusion(k=10)+BinarizedAUs | 0.12            | 94.09  | 0.0379    | 0.0248 | 0.0397 | -       | 49        |
| Ours                         | 0.26            | 86.73  | 0.1160    | 0.0346 | 0.1162 | 81.28       | 46.96     |

And on the test set, it is as follows:
|                              | Appropriateness |        | Diversity |        |        | Realism | Synchrony |
|------------------------------|:---------------:|:------:|:---------:|:------:|:------:|:-------:|:---------:|
|                              | FRC             | FRD    | FRDvs     | FRVar  | FRDiv  | FRRea   | FRSyn     |
| Ours                         | 0.3190 |	82.0288 |	0.1165 |	0.0344 |	0.1162	| 34.6685 |	43.0918|

### Output comparison with Learning2Listen's baseline:
<video src="submit_videos/comparison_with_learning2listen.mp4" controls></video>

## Prequisites

The code is developed and tested only with Python 3.10 in Ubuntu 22.04.

## Installation
This project is developed with a dependency on PyTorch, PyTorch3D, lightning, and Transformers. The installation is as follows:

**To visualize the output please have ffmpeg installed in your system.**

1. Clone the repository.

2. Create a conda virtual environment and activate the environment:
    - `conda create -n aisreact2024 python=3.10`
    - `conda activate aisreact2024`

2. Install the CUDA in conda environment, this is for rebuild pytorch-3d, if your pytorch-3d is already installed, you can skip this step. The command is as follows:
    `conda install cuda -c nvidia`


3. Install other dependencies from requirements.txt:
    - `pip install -r requirements.txt`

4. Restart the terminal and activate the conda environment again.

4. Install PyTorch-3D with CUDA:
    - `FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@8772fe0de8b9332be501bf0e61bd5502e3075e60"`

We noticed that the installation of PyTorch-3D with CUDA is not straightforward, sometimes it will cause an error, please follow the [Pytorch-3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to solve the problem. 


## Download checkpoints, external and dataset

1. Download the [checkpoints and external tool](https://drive.google.com/drive/folders/1hhBThohHAdDT1shZPyxzmsnMSEoI6J07?usp=sharing) and put it to the checkpoint and external folder separately.
2. Please refer to the [REACT2024](https://sites.google.com/cam.ac.uk/react2024/home) to download the dataset.
3. You can run the demo without the dataset.

## Model input and output

1. Input: 
    - Speaker's voice: path to the audio file as the wav2vec will process the feature.

    - Speaker's face: a tensor 58D containing 3DMM features.

2. Output:
    - A tensor 58D containing 3DMM features of the listener's face.

    - A tensor 25D emotion features of the listener's face.

## Dataloader

The dataloader is the same as the original dataloader, except that instead of loading MFCC as audio features, we change it to load the path to the audio file with the start and end time of the audio segment. This is because we use wav2vec2 to process the audio feature. 

## Training

Training is divided into 2 stages:
1. Train the tokenizer by running `p1_face_tokenizer_pretrain.py`. Change the config in the main function for the model config parameters and training hyperparameters.  
2. Train the reaction predictor by running `p2_reaction_predictor.py`. Change the config in the main function for the model config parameters and training hyperparameters. Make sure that the path to the desired tokenizer is correct and its config is the same as its training one, or it will cause a loading error.

This code tracks the model version and config by pushing all hyperparameters to wandb. You can switch to TensorBoard by commenting out logging to wandb and adding TensorBoard instead.

In case of resuming training, you can specify the `resume_checkpoint` in the main function of the training script and set the argument `resume` to `True`.

### Command
Please change the path to the dataset and place the checkpoint in the correct path before running the command.

1. Face Tokenizer training, after training it will save the checkpoint in the `output_dir`:
```bash
python p1_face_tokenizer.py --run_name "face_tokenizer" --quantize_type "fsq" --output_dir "output/face_tokenizer" --dataset_path "/home/usr/react2024/data"
```

2. Reaction Predictor training, after training it will save the checkpoint in the `output_dir`:
```bash
python p2_react_predictor --output_dir output --tokenizer_checkpoint checkpoint/face_tokenizer_epoch=185-step=37200.ckpt --dataset_path /home/usr/react2024/data
```

**To see the training commands, please refer to the `run.bash` file.**

## Metrics harvesting

After each training section, it will collect predictions and compute the metrics. 

You can turn off training by specifying `test=True` in the main parameter and input the `test_checkpoint` to run the model on the test/validation set by setting the `collect_metrics_in` to `test` or `val` in the `p2_reaction_predictor.py` file. 

The script will visualize the predictions first, then run multiple `test_dataloaders` to collect the metrics for diversity. Note that if the data config `submit_video_only` is `True`, then the script will only render the 4 necessary submission videos. And if `submit_video_only` is `False`, then the script will render every `render_skip_step` in `predictor_config`.

You can control test output by modify the test in the end of the main function of `p2_reaction_predictor.py`.

### Command

A sample command to run the test is as follows, please change the path to the dataset and place the checkpoint in the correct path before running the command:
```bash
python p2_react_predictor.py --test --test_checkpoint checkpoint/react_predictor_epoch=91-step=9200.ckpt --tokenizer_checkpoint checkpoint/face_tokenizer_epoch=185-step=37200.ckpt --output_dir output --dataset_path /home/usr/react2024/data
```

**To see the test command, please check the `run.bash` file.**

## Visualization

The visualization code is contain in `test_step` function of `p2_reaction_predictor.py`. There will be folders of frame generated for the video and the video itself. The video will be generated by the `ffmpeg` command inside. So that, when run the test function it will automatic render the validation require video. The visualization folder is structured as

```
output_dir
├── run_name                # this is to denote training time
    ├── infer_name          # this is to denote the inference time
        ├── val
            ├── only_submit_video   
                ├── mesh_video_b*_ind* 
                # contain the frame of the mesh predicted
                    ├── frame_0.jpg
                    ├── frame_1.jpg
                    ├── ....
                    ├── frame_750.jpg
                ├── fake_video_b*_ind* 
                # contain the frame of the photo-rendered predicted
                ├── b*_ind*.json 
                # contain the address of frames for each video, can be used by `render_final_video.py` to render the video
                ├── b*_ind*.mp4 
                # video of speaker, listener, mesh, and photo-rendered from left to right  
            ├── render_for_fid    
                ├── mesh_video_b*_ind* 
                # contain the frame of the mesh predicted
                ├── fake_video_b*_ind* 
                # contain the frame of the photo-rendered predicted
                ├── b*_ind*.json 
                # contain the address of frames for each video, can be used by `render_final_video.py` to render the video
                ├── listener_gt.pt  
                # the ground truth listener's emotion vector
                ├── speaker_gt.pt   
                # the ground truth speaker's emotion vector
                ├── listener_pred.pt 
                # the predicted listener's emotion vector
                ├── metrics.txt 
                # the computed-metrics for the dataset
        ├── test            

```

## Data

We have kept the data folder structure the same as the react2024 dataset.
```data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files```

The example of data structure.
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.jpg
                       ├── ....
                       ├── 751.jpg
                   ├── ....
           ├── ....
       ├── RECOLA
       ├── UDIVA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
            
```
 
- The task is to predict one role's reaction ('Expert' or 'Novice',  'P25' or 'P26'....) to the other ('Novice' or 'Expert',  'P26' or 'P25'....).
- 3D_FV_files involve extracted 3DMM coefficients (including expression (52 dim), angle (3 dim) and translation (3 dim) coefficients).
- The frame rate of processed videos in each site is 25 (fps = 25), height = 256, width = 256. And each video clip has 751 frames (about 30s), The samping rate of audio files is 44100. 
- The csv files for baseline training and validation dataloader are now avaliable at 'data/train.csv' and 'data/val.csv'


## Known issues
- Not yet fixed: `scale_dot_product_attention` might produce NaN when encountering an all-false mask on old CUDA devices, which can lead to test failures. 

## Acknowledgements
Thank @learning2listen, Lucidrains, and @nano-gpt for their open-source projects. Many parts of this project were originally forked from their repositories.
