GLiT: Neural Architecture Search for Global and Local Image Transformer
=========================================
Code for [GLiT: Neural Architecture Search for Global and Local Image Transformer](https://arxiv.org/abs/2107.02960) accepted by ICCV2021


## Requirements
- torch==1.7.0
- torchvision==0.8.1
- timm==0.3.2

The requirements.txt file lists other Python libraries that this project depends on, and they will be installed using:
pip3 install -r requirements.txt

## Results
|method|  FLOPs    |   Params |   Top-1   | model|
|:------:|:----------:|:----------:|:----------:|:----------:|
|GLiT-Tiny| 1.33G | 7.3M | 76.4 | [Glit_Tiny](https://drive.google.com/file/d/1ryxn9TEwnoDTTQxv5JMyWpvU2OuOMLqL/view?usp=sharing)|


## Training
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model glit_tiny_patch16_224 --clip-grad 1.0 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

## Evaluation
Download the model [Glit_Tiny](https://drive.google.com/file/d/1ryxn9TEwnoDTTQxv5JMyWpvU2OuOMLqL/view?usp=sharing)
```
python main.py --eval --resume glit_tiny.pth.tar --data-path /path/to/imagenet

```

## Thanks
This implementation is based on [DeiT](https://github.com/facebookresearch/deit). Please ref to their reposity for more details.

## Citation
If you find that this project helps your research, please consider citing our paper:
```
@inproceedings{chen2021glit,
  title={Glit: Neural architecture search for global and local image transformer},
  author={Chen, Boyu and Li, Peixia and Li, Chuming and Li, Baopu and Bai, Lei and Lin, Chen and Sun, Ming and Yan, Junjie and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12--21},
  year={2021}
}
```
