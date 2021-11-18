# Late Fusion-based Video Transformer for Facial Micro-expression Recognition
This implementation is ans official code on the paper *Late Fusion-based Video Transformer for Facial Micro-expression Recognition*. In paper, we just validated one micro-facial dataset, SMIC. But we implemented our code to be able to expriment on the CASME2 and SAMM dataset for futher research.

## Requirements
* Python 3.9.6
* Pytorch 1.9.0
* TorchVision 0.10.0
Details are specified in `requirements.txt`. Use `pip install -r requirements.txt`. Please be careful to install the pytorch. We don't test all the version of pytorch. 


## Prepare for Learning
In SMIC, there is no version of face crop. The upper body and background other than the face are not very related to facial micro-expressions, so they should be removed for learning. Then we used [MTCNN](https://github.com/timesler/facenet-pytorch) for face cropping. In addition, we also implemented code using [MobilenetV2](https://github.com/cunjian/pytorch_face_landmark) to obtain face alignment for futher research.

### Usage
You can prepare dataset by using `prepare.py`.
```
python3 prepare.py -h
usage: prepare.py [-h] [--dir DIR] [--gpu GPU] [--ext EXT]

optional arguments:
  -h, --help  show this help message and exit
  --dir DIR   Dataset directory. Ex) data/SMIC
  --gpu GPU   GPU number to use. If not available, use CPU
  --ext EXT   Choose image file extention in dataset
```

For cropping and face alignment, use the following command:
```
python3 prepare.py --dir /path/data/smic --gpu 0 --ext jpg
```
Cropped images and landmarks numpy files will be saved in the following directories separately: 

> Dataset path: `/path/data/smic`
> * Cropped images path: `/path/data/smic_CROP`
> * Landmarks path: `/path/data/smic_LANDMARK`

## Training and Validation
You can train and validate the models in paper by using `main.py`.
```
python3 main.py -h
usage: main.py [-h] [-tb TENSOR_BOARD] [-imgsize IMG_SIZE [IMG_SIZE ...]]
               [-feature {RGB,GRAY,GRAY_OF,ONLY_OF,RGB_OF}] [-interpol {linear,tim}] [-num_frames NUM_FRAMES]
               [-bs BATCH_SIZE] [-lr LR] [-ep EPOCH] [-gpu GPU [GPU ...]] [--imbalanced_sampler] [--subtb]
               MODEL DATA_PATH DATASET

positional arguments:
  MODEL                 Like ch{Num}_{Backbone}. You can choose {Num} in [1, 3, 4, 6] and {Backbone} in [resnext,
                        swin, timesformer].
  DATA_PATH             Dataset directory except dataset name. Ex) /home/data not /home/data/SMIC
  DATASET               Choose your dataset in ['SMIC', 'CASME2', 'SAMM']

optional arguments:
  -h, --help            show this help message and exit
  -tb TENSOR_BOARD      Tensorboad dir name
  -imgsize IMG_SIZE [IMG_SIZE ...]
                        1 int or 2 int like 120(H) 64(W). For 1 int, resize width and height same
  -feature {RGB,GRAY,GRAY_OF,ONLY_OF,RGB_OF}
                        Choose image mode or use OF with grayscale
  -interpol {linear,tim}
                        Choose using interpolation for fixed sequence length
  -num_frames NUM_FRAMES
                        Video length in frames
  -bs BATCH_SIZE        Train batch size
  -lr LR                Learning rate
  -ep EPOCH             Num of epochs
  -gpu GPU [GPU ...]    GPU number to use. If not available, use CPU
  --imbalanced_sampler  Flag to determine whether to use imbalanced sampler for balancing dataset with respect to class
  --subtb               Flag to determine whether to track each subject metric with Tensorboard
```


For training and validation the paper's proposed model, use the following command:
```
python3 main.py ch3_3_fusion_cnn_like /path/data SMIC -imgsize 224 -feature GRAY_OF -interpol linear -num_frames 16 -bs 1 -ep 100 --imbalanced_sampler 
```