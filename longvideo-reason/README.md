---
dataset_info:
  features:
  - name: problem_id
    dtype: int64
  - name: problem
    dtype: string
  - name: data_type
    dtype: string
  - name: problem_type
    dtype: string
  - name: reasoning
    dtype: string
  - name: videos
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 111619065
    num_examples: 51200
  - name: test
    num_bytes: 2041214
    num_examples: 1000
  - name: validation
    num_bytes: 127216
    num_examples: 62
  - name: openended
    num_bytes: 16075996
    num_examples: 52800
  download_size: 69569967
  dataset_size: 129863491
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
  - split: openended
    path: data/openended-*
---
<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/NVlabs/Long-RL/main/assets/long-rl-logo.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Scaling RL to Long Sequences (Training, Validation and Test Dataset - for research only)

[![Paper](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/abs/2507.07966)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/NVlabs/Long-RL/blob/main/LICENSE)

<div align="center">

[![Watch the video](https://raw.githubusercontent.com/NVlabs/Long-RL/main/assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Data Distribution

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/NVlabs/Long-RL/main/assets/data_distribution.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

We strategically construct a high-quality dataset with CoT annotations for long video reasoning, named LongVideo-Reason. Leveraging a powerful VLM (NVILA-8B) and a leading open-source reasoning LLM, we develop a dataset comprising 52K high-quality Question-Reasoning-Answer pairs for long videos. We use 18K high-quality samples for Long-CoT-SFT to initialize the model's reasoning and instruction-following abilities, and 33K samples with an additional 110K video data for reinforcement learning. This two-stage training combines high-quality reasoning annotations with reinforcement learning, enabling LongVILA-R1 to achieve superior and generalized video reasoning. We also manually curate a balanced set of 1K long-video samples to build a new benchmark, LongVideo-Reason-eval, that evaluates performance from four perspectives: Temporal, Goal and Purpose, Spatial, and Plot and Narrative, for a comprehensive assessment.


**LongVideo-Reason (Train, 52k) [[Data Link](https://huggingface.co/datasets/LongVideo-Reason/longvideo-reason)]**

**LongVideo-Reason-eval (Test, 1k) [[Data Link](https://huggingface.co/datasets/LongVideo-Reason/longvideo_eval_videos)]**


## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```
If you want to train Qwen-Omni models, please
```bash
bash vllm_replace.sh
```

## Training
### Single node
For single node (within 8 GPUs), you can refer to the training scripts in the `examples` directory. For example,
```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes
For jobs that requires multi-nodes, you can refer to the ways mentioned in the EasyR1 repo, [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

We provide additional examples for `sbatch` scripts like, where `TRAIN_SCRIPT` is the script to train on single node, `NNODES` is the number of nodes required.
```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For example, 
```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format
This follows the ways in the EasyR1 repo.
```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## Evaluation
We provide the instruction on evaluating models on our `LongVideo-Reason` benchmark in the `eval` [directory](https://github.com/NVlabs/Long-RL/tree/main/eval). 



## How to contribute
- Make sure to have git installed.
- Create your own [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
- Clone the repository on your local machine, using git clone and pasting the url of this project.
- Read both the `Installation` sections above.
- Commit and push your changes.
- Make a pull request when finished modifying the project.


## Core Contributors
[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all possible contributions and will acknowledge all contributors clearly.

## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.

```bibtex
@misc{long-rl,
  title = {Long-RL: Scaling RL to Long Sequences},
  author = {Yukang Chen, Wei Huang, Shuai Yang, Qinghao Hu, Baifeng Shi, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu,Hongxu Yin, Yao Lu, Song Han},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVlabs/Long-RL}},
}
```
```bibtex
@article{chen2025longvila-r1,
      title={Scaling RL to Long Videos},
      author={Yukang Chen and Wei Huang and Baifeng Shi and Qinghao Hu and Hanrong Ye and Ligeng Zhu and Zhijian Liu and Pavlo Molchanov and Jan Kautz and Xiaojuan Qi and Sifei Liu and Hongxu Yin and Yao Lu and Song Han},
      year={2025},
      eprint={2507.07966},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
@inproceedings{chen2024longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      booktitle={The International Conference on Learning Representations (ICLR)},
      year={2025},
}
```

## Acknowledgement
- [EasyR1](https://github.com/hiyouga/EasyR1): the codebase we built upon. Thanks for their wonderful work.
- [verl](https://github.com/volcengine/verl): the RL training framework we built upon.
- [vllm](https://github.com/vllm-project/vllm): we built upon vllm for the rollout engine.
- [Flow-GRPO](https://github.com/yifan123/flow_grpo): we refer to the Flow-GRPO for the image/video generation RL part.
- [Shot2story](https://arxiv.org/abs/2312.10300): we curate 18K long videos from the Shot2Story.

