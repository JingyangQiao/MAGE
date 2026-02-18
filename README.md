# MAGE
Official Pytorch implementation for Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework

Paper will come soon!

## 1. Abstract
Dual-to-Dual MLLMs refer to Multimodal Large Language Models, which canenable unified multimodal comprehension and generation through text and imagemodalities. Although exhibiting strong instantaneous learning and generalizationcapabilities, Dual-to-Dual MLLMs still remain deficient in lifelong evolution, sig-nificantly affecting continual adaptation to dynamic real-world scenarios. One ofthe challenges is that learning new tasks inevitably destroys the learned knowl-edge. Beyond traditional catastrophic forgetting, Dual-to-Dual MLLMs face otherchallenges, including hallucination, instruction unfollowing, and failures in cross-modal knowledge transfer. However, no standardized continual learning frameworkfor Dual-to-Dual MLLMs has been established yet, leaving these challenges unex-plored. Thus, in this paper, we establish Continual-NExT, a continual learningframework for Dual-to-Dual MLLMs with deliberately-architected evaluation met-rics. To improve the continual learning capability of Dual-to-Dual MLLMs, wepropose an efficient MAGE (Mixture and Aggregation of General LoRA andExpert LoRA) method to further facilitate knowledge transfer across modalitiesand mitigate forgetting. Extensive experiments demonstrate that MAGE outper-forms other continual learning methods and achieves state-of-the-art performance.

## 2. Continual-NExT Dataset
1. Please download the Continual-NExT datasets **(annotations)** in our huggingface page (https://huggingface.co/datasets/jingyang/Continual-NExT). Then save these datasets in the `./data` dictionary.

2. Please download the Continual datasets **(images)** according to the following Table:

|  Dataset  | Download Path  |
|  :----:  | :----:  |
| VQAv2 | [train2014](http://images.cocodataset.org/zips/train2014.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip) |
| ImageNet | [images](https://image-net.org/challenges/LSVRC/index.php) |
| OCR-VQA  | [images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_) |
| GQA | [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) |
| RefCOCO(Grounding) | [train2014](http://images.cocodataset.org/zips/train2014.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip) |
| HQEdit | [images](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit) |

3. Then, organize the instructions as follows:

```
├── data
|   └── SEED-Data-VQAv2
|       └── COCO2014
|    	    └── train2014
|           └── val2014
|       └──annotations
|       └──questions
|   └── SEED-Data-ImageNet
|       └── imagenet
|    	    └── train
|           └── val
|       └──annotations
|       └──questions
|   └── SEED-Data-Fliackr30k
|       └── images
|       └──annotations
|       └──questions
|   └── SEED-Data-OCRVQA
|       └── OCR-VQA
|       └──annotations
|       └──questions
|   └── SEED-Data-Grounding
|       └── COCO2014
|    	    └── train2014
|       └──annotations
|       └──questions
|   └── SEED-Data-HQEdit
|       └── images
|    	    └── source_images
|           └── target_images
|       └──annotations
|       └──questions
```

## 3. Install Repository
1. Clone this repository
``` 
git clone https://github.com/JingyangQiao/MAGE.git
cd MAGE
```
2. Install Package
```
conda create -n mage python=3.10 -y
conda activate mage
pip install -r requirements.txt
```

## 4. Checkpoints
1. Please prepare the pre-trained SEED-X checkpoints as the instructions in [SEED-X](https://github.com/AILab-CVC/SEED-X).

2. In order to ensure that each CL (Continual Learning) baseline has a common initial training checkpoint, we did not adopt any CL methods. Instead, we directly train on the initial task (VQAv2) and obtain the LoRA weights and Mage (MoELoRA) agent weights, which can be downloaded in [LoRA](https://huggingface.co/jingyang/Continual-NExT-VQAv2/tree/main/lora) and [Mage](https://huggingface.co/jingyang/Continual-NExT-VQAv2/tree/main/mage). Finally, these checkpoints needed to be stored as `./train_output/mage/VQAv2/pytorch_model.bin`

3. Then, organize the checkpoints as follows:

```
├── pretrained
|   └── clip
|   └── QwenViT
|   └── seed_x_i
|   └── seed_x_edit
|   └──stable-diffusion-xl-base-1.0
```

## 5. Training
1. We provide the training scripts of tasks (i.e., VQAv2, ImageNet, Flickr30k, OCRVQA, RefCOCO) in `./scripts/train/train_seed_x_sft_comp_gen.sh`.
We provide the training scripts of tasks (i.e., HQEdit) in `./scripts/train/train_seed_x_sft_edit.sh`.

2. Uncomment the current training dataset and comment out the other datasets in the file `./configs/data/sft_comprehension_gen.yaml`.

3. In `./configs/clm_models/agent_seed_x_i.yaml`, replace the `pretrained_model_path` variable with the path of agent weights obtained from the previous task's training.

4. You can run the following commands for training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/train/train_seed_x_sft_comp_gen.sh
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/train/train_seed_x_sft_edit.sh
```

## 6. Saving

1. The saved checkpoints will in: `./train_output`.

2. Change the content of the `./train_output/seed_x_sft_comp_gen/checkpoint-xxxx/zero_to_fp32.py` or `./train_output/seed_x_sft_edit/checkpoint-xxxx/zero_to_fp32.py` with `./train_output/zero_to_fp32.py`.

3. Operate as the instructions of processing saved checkpoints after training in [SEED-X](https://github.com/AILab-CVC/SEED-X).

4. The saved agent checkpoints will in `./train_output/seed_x_sft_comp_gen/checkpoint-xxxx/pytorch_model.bin/pytorch_model.bin` or in `./train_output/seed_x_sft_edit/checkpoint-xxxx/pytorch_model.bin/pytorch_model.bin`.

## 7. Evaluation
1. In `./configs/clm_models/agent_seed_x_i.yaml`, replace the `pretrained_model_path` variable with the path of agent weights which you want to evaluate.

2. You can run the following commands for evaluation:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/CoIN/Eval/eval_all.sh
```
3. The evaluation results will be saved in `./results`.

## License
This repository is released under the MIT license.

## Citation
```markdown
@inproceedings{mage,
  title={Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework},
  author={Jingyang Qiao and Zhizhong Zhang and Xin Tan and Jingyu Gong and Yanyun Qu and Yuan Xie},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgement
[SEED-X](https://github.com/AILab-CVC/SEED-X): the codebase we built upon.
