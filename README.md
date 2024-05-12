# AGNR
The codes for the work "Attention-Guided and Noise-Resistant Learning for Robust Medical Image Segmentation".  We updated the Reproducibility. I hope this will help you to reproduce the results.

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
python train.py --dataset Synapse --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR   --img_size 224 --base_lr 0.05 --batch_size 24

```

- Test 

```bash
python test.py --dataset Synapse --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml  --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --item_id your ITEM_ID
```



## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)



