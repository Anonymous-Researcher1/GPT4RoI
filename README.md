# GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest :fire: [Demo](http://139.196.83.164:7000/) :fire:




<p align="center">
    <img src="figs/demo1.gif" width="80%"> <br>
  <p align="center" style="font-size:1.2vw;">Single-Region Understanding</p>
</p>
<p align="center">
    <img src="figs/demo2.gif" width="80%"> <br>
  <p align="center"  style="font-size:1.2vw;">Multiple-Region Understanding</p>
</p>



## Introduction
<p align="center">
    <img src="figs/framework.png" width="70%"> <br>
</p>



## Updates

- All training and inference code has been released, you can try demo [here](http://139.196.83.164:7000/) :fire::fire::fire:


## Contents
- [Install](#Install)
- [Data](#Data)
- [Training](#Training)
- [Gradio](#Gradio)
- [Acknowledge](#Acknowledge)


## Install
1. Clone the `GPT4RoI`
```python
git clone https://github.com/Anonymous-Researcher1/GPT4RoI
cd gpt4roi
```

2. Create the env
```shell
conda create -n gpt4roi python=3.10 -y
conda activate gpt4roi
pip install --upgrade pip  # enable PEP 660 support
pip install setuptools_scm
pip install --no-cache-dir  -e .
# please use conda re-install the torch, pip may loss some runtime lib
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
```
3. Install the `flash-attn` package 
```
pip install ninja
pip install flash-attn --no-build-isolation
```
4. install the `mmcv-1.4.7` package
Make sure that your `nvcc -V` is consistent with cudatookit version of `python -c "import torch;print(torch.version.cuda)`.
```shell
cd mmcv-1.4.7
MMCV_WITH_OPS=1 pip install -e .
```


## Data

Our dataset includes RefCOCO, RefCOCO+, RefCOCOg, Visual Genome, Flickr30K entities, and the VCR dataset. We are sincerely grateful to the creators of these datasets, especially for the VCR dataset, for their forward-thinking in creating these dataset.

The dataset section of this repository may appear somewhat messy, especially the VCR part(still finishing), which may cause GPT4RoI not be very user-friendly. We are currently working on formulating the datasets into a unified format and will be accompanying them with stronger models. Please stay tuned for updates.


You can download the corresponding dataset from the official website and organize it as follows. Afterwards, you can modify the ```gpt4roi/configs/dataset_config.json``` file to select the specific dataset you want to use:

```text
GPT4RoI
├── data
│   ├── coco_det
│   │   ├── annotations
│   │   │      ├──instances_train2017.json
│   │   ├── train2017/
│   ├── mdetr_annotations
│   │          ├──finetune_refcoco_train.json
│   │          ├──finetune_refcoco+_train.json
│   │          ├──finetune_refcocog_train.json
│   │          ├──final_flickr_mergedGT_train.json
│   ├── coco_imgs/
│   ├── flickr30k-images/
│   ├── visual_genome
│   │          ├──train.json
│   │          ├──vg_all/
│   ├── llava
│   │   ├── llava_instruct_150k.json
│   │   ├── llava_150k_bbox_pred_results.pkl
│   ├── vcr
│   │   ├── train.jsonl
│   │   ├── vcr1images/
```
### NOTE
1. coco_imgs should contains all coco image(you can soft link them to this directory.
2. We use Visual_Genome_Dataset_V1.2, available for download from  [OpenDataLab](https://opendatalab.com/). Ensure to download the  [train.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/train.json), you should create a soft link for all VG images to the directory `vg_all`.


## Training
GPT4RoI is trained on 8 A100 with the following code.

### STAGE 1
Vicuna-v0, an instruction-tuned chatbot, is the base model for this setup. In order to prepare it, first download the delta weights available [here](https://huggingface.co/lmsys/vicuna-7b-delta-v0). To obtain the original weights, follow the instructions provided [here](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#how-to-apply-delta-weights-for-weights-v11-and-v0) to integrate these delta weights into LLaMA-7B.

Ensure to download the following projector weight file: [LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin](https://huggingface.co/liuhaotian/LLaVA-Pretrained-Projectors/resolve/main/LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin).

Additionally, you have the flexibility to choose from different versions of Vicuna (such as the 13B version or llama v2 chatbot) and the corresponding projector weights from [LLaVA](https://github.com/haotian-liu/LLaVA) to meet your specific requirements effectively.
`exp/stage1` is the work directory. 
```Shell
bash train_stage1.sh exp/stage1
# Resume training in stage1
# bash train_stage1.sh exp/stage1
```
### STAGE 2

`exp/stage2` is the work directory. and you should give the work directory of stage1 so we can load the corresponding weight as pretrain model.
```Shell
# At the beginning of stage2
bash train_stage2.sh exp/stage2 exp/stage1
# Resume training in stage2
# bash train_stage2.sh exp/stage2 
```



## Gradio
Please install [Gradio Box](https://github.com/ShoufaChen/gradio-dev) first.
```python
python gpt4roi/app.py
```
### NOTES
1. ```prompt format in GPT4RoI```
You should always use `<region1>, <region2>...` to refer the new bounding box in the image when you first draw them. Then you can use normal `region 1` in the conversation to refer the instance.
2. You should always click the `clear all` buttul and waiting the clear process finished before you start a new conversation.

<p align="center">
    <img src="figs/fig1.png" width="100%"> <br>
<p align="center"  style="font-size:1.2vw;">Multiple Rounds of Dialogue</p>
</p>




## Acknowledge

- [LLaVA](https://github.com/haotian-liu/LLaVA): The codebase we built upon.
- [Vicuna](https://github.com/lm-sys/FastChat): The LLM we used.
- [VCR](https://visualcommonsense.com/): We get strong region reasoning ability from this forward thinking dataset.