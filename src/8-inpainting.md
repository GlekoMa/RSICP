**To run inpainting, you need check out the [lama repo ](https://github.com/advimman/lama) README first!**

Make shure you are in lama folder

```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

You need to prepare following image folders:

```
$ ls my_dataset
train
val_source # 2000 or more images
visual_test_source # 100 or more images
eval_source # 2000 or more images
```

```
$ ls my_dataset/val
image1_mask000.png
image1.png
image2_mask000.png
image2.png
...
```

Generate location config file which locate these folders:

touch my_dataset.yaml
echo "data_root_dir: $(pwd)/my_dataset/" >> my_dataset.yaml
echo "out_root_dir: $(pwd)/experiments/" >> my_dataset.yaml
echo "tb_dir: $(pwd)/tb_logs/" >> my_dataset.yaml
mv my_dataset.yaml ${PWD}/configs/training/location/

Check data config for consistency with my_dataset folder structure:

```
$ cat ${PWD}/configs/training/data/my_dataset
...
train:
  indir: ${location.data_root_dir}/train
  ...
val:
  indir: ${location.data_root_dir}/val
  img_suffix: .png
visual_test:
  indir: ${location.data_root_dir}/visual_test
  img_suffix: .png
```

Run training

```
python3 bin/train.py -cn lama-fourier location=my_dataset data.batch_size=10
```

Evaluation: LaMa training procedure picks best few models according to 
scores on my_dataset/val/ 

To evaluate one of your best models (i.e. at epoch=32) 
on previously unseen my_dataset/eval do the following
for 'bbox_mask', 'filter_mask' and 'combine_mask':

infer:

```
python3 bin/predict.py \
model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
indir=$(pwd)/my_dataset/eval/random_<size>_512/ \
outdir=$(pwd)/inference/my_dataset/random_<size>_512 \
model.checkpoint=epoch32.ckpt
```

metrics calculation:

```
python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/my_dataset/eval/random_<mask_type>_800/ \
$(pwd)/inference/my_dataset/random_<mask_type>_800 \
$(pwd)/inference/my_dataset/random_<mask_type>_800_metrics.csv
```
