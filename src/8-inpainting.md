**To run inpainting, you need check out the [lama repo](https://github.com/advimman/lama) README first!**

Make shure you are in lama folder

```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

Then run

```
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/<our_dataset> outdir=$(pwd)/output
```

The `output` directory would contain the metrics.
