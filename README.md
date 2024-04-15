Complementary Trilateral Decoder for Fast and Accurate Salient Object Detection

### Installation

```
conda create -n PyTorch python=3.11
conda activate PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.5.5.64
pip install tqdm
pip install timm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for Single-GPU training
* Run `bash main.sh $ --train` for Multi-GPU training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Demo

* Configure your image path in `main.py` for visualizing the demo
* Run `python main.py --demo` for demo

### Results

![Alt Text](./demo/demo.jpg)

