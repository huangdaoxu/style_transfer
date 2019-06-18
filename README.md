# style transfer with deep learning
![APM](https://img.shields.io/apm/l/github.svg)

## Requirements
python == 3.6<br>
tensorflow == 1.10.0<br>
numpy == 1.14.5<br>

## Usage
**Frist**, download all dependent files.<br>

Please download slim vgg_16 check point from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz<br>
Please download coco dataset from http://images.cocodataset.org/zips/val2017.zip<br>
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
wget http://images.cocodataset.org/zips/val2017.zip
```

**Second**, fit your transfer model.
```
python train.py --vgg_path "your vgg pre-trained path" --model_path "your transfer model path"
--style_image_path "choose one style image" --train_dataset "coco dataset path" --epoch 20
--batch_size 4 --learning_rate 0.001 --content_loss_weight 1.0 --style_loss_weight 100.0 --summary_path "summary path"
```

**Third**, you can use tensorboard to visual training process. you should access http://127.0.0.1:8888
```
tensorboard --logdir "your summary path" --port 8888
```

**Last**, you can use trained model to inference.
```
python inference.py --test_image "test image" --model_path "your transfer model path" --saved_path "transfromed image path"
```

## Results

### Style image
![](https://github.com/huangdaoxu/style_transfer/blob/master/style_images/starry.jpg)
![](https://github.com/huangdaoxu/style_transfer/blob/master/style_images/wave.jpg)

### The original image
![](https://github.com/huangdaoxu/style_transfer/blob/master/test/sunshine_boy.jpeg)

### Transformed image
![](https://github.com/huangdaoxu/style_transfer/blob/master/test/inference_starry.jpg)
![](https://github.com/huangdaoxu/style_transfer/blob/master/test/inference_wave.jpg)
![](https://github.com/huangdaoxu/style_transfer/blob/master/test/inference_landscape.jpg)
