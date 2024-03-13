# CNNのサンプルプログラム
CNNベースの手法による画像分類のサンプルコード

## 動作環境
<details>
<summary>ライブラリのバージョン</summary>
 
* cuda 12.1
* python 3.6.9
* torch 1.8.1+cu111
* torchaudio  0.8.1
* torchinfo 1.5.4
* torchmetrics  0.8.2
* torchsummary  1.5.1
* torchvision 0.9.1+cu111
* timm  0.5.4
* tlt  0.1.0
* numpy  1.19.5
* Pillow  8.4.0
* scikit-image  0.17.2
* scikit-learn  0.24.2
* tqdm  4.64.0
* opencv-python  4.5.1.48
* opencv-python-headless  4.6.0.66
* scipy  1.5.4
* matplotlib  3.3.4
* mmcv  1.7.1
</details>

## ファイル＆フォルダ一覧

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|
|----|----|
|vgg_train.py|VGGを学習するコード．|
|resnet_train.py|ResNetを学習するコード．|
|efficientnet_train.py|EfficientNetを学習するコード(EfficientNetはモデルによって画像サイズが異なるので適宜リサイズをしてください)．|
|convnext_train.py|ConvNeXtを学習するコード．|
|trainer.py|学習ループのコード．|
|make_graph.py|学習曲線を可視化するコード．|
</details>

## 実行手順

### 環境設定

[先述の環境](https://github.com/SyunkiTakase/CNN_Classification_Sample?tab=readme-ov-file#%E5%8B%95%E4%BD%9C%E7%92%B0%E5%A2%83)を整えてください．

### 学習
ハイパーパラメータは適宜調整してください．

<details>
<summary>VGG，ResNet，EfficientNet，ConvNeXtのファインチューニング(CIFAR-10)</summary>

VGGの学習 
```
python3 vgg_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10
```
ResNetの学習
```
python3 resnet_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10
```
EfficientNetの学習
```
python3 efficientnet_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10
```
ConvNeXtの学習
```
python3 convnext_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10
```
</details>

<details>
<summary>VGG，ResNet，EfficientNet，ConvNeXtのファインチューニング(CIFAR-100)</summary>
 
VGGの学習 
```
python3 vgg_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100
```
ResNetの学習
```
python3 resnet_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100
```
EfficientNetの学習
```
python3 efficientnet_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100
```
ConvNeXtの学習
```
python3 convnext_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100
```
</details>

## 参考文献
* 参考にした論文
  * VGG
    * Very Deep Convolutional Networks for Large-Scale Image Recognition
  * ResNet
    * Deep Residual Learning for Image Recognition
  * EfficientNet
    * EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
  * ConvNeXt
    * A ConvNet for the 2020s

* 参考にしたリポジトリ 
  * timm
    * https://github.com/huggingface/pytorch-image-models
  * VGG
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vgg.py
  * ResNet
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
  * EfficientNet
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
  * ConvNeXt
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
