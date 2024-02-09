## Outlier Detection

Will update
![Outlier Detection](https://github.com/danielgural/semantic_video_search/blob/main/assets/video_semantic_search.gif)

This plugin is a Python plugin that allows for you to find the outliers in your dataset!

🔎 Find those pesky outliers in just a few minutes!

## Installation

```shell
fiftyone plugins download https://github.com/danielgural/outlier_detection
```

## Operators

### `outlier_detection`

Finds outliers in your dataset based on embeddings from [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/models.html). Choose which algorithm you want to use to find outliers in your dataset. 

Make sure to take advantage of optional inputs such as finding outliers within a specific class, tagging outliers, or optionally regenerating embeddings from a different model!

![Outlier Detection Inputs](https://github.com/danielgural/outlier_detection/blob/main/assets/outlier_inputs.png)

Currently only [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) is supported, but more will come soon and feel free to contribute your own!



### Supported Embedding Models

| Model Name                                                | Framework   |
|-----------------------------------------------------------|-------------|
| alexnet-imagenet-torch                                    | Torch       |
| classification-transformer-torch                          | Torch       |
| clip-vit-base32-torch                                     | Torch       |
| densenet121-imagenet-torch                                | Torch       |
| densenet161-imagenet-torch                                | Torch       |
| densenet169-imagenet-torch                                | Torch       |
| densenet201-imagenet-torch                                | Torch       |
| detection-transformer-torch                               | Torch       |
| dinov2-vitb14-torch                                       | Torch       |
| dinov2-vitg14-torch                                       | Torch       |
| dinov2-vitl14-torch                                       | Torch       |
| dinov2-vits14-torch                                       | Torch       |
| googlenet-imagenet-torch                                  | Torch       |
| inception-resnet-v2-imagenet-tf1                          | TF1         |
| inception-v3-imagenet-torch                               | Torch       |
| inception-v4-imagenet-tf1                                 | TF1         |
| mobilenet-v2-imagenet-tf1                                 | TF1         |
| mobilenet-v2-imagenet-torch                               | Torch       |
| open-clip-torch                                           | Torch       |
| resnet-v1-50-imagenet-tf1                                 | TF1         |
| resnet-v2-50-imagenet-tf1                                 | TF1         |
| resnet101-imagenet-torch                                  | Torch       |
| resnet152-imagenet-torch                                  | Torch       |
| resnet18-imagenet-torch                                   | Torch       |
| resnet34-imagenet-torch                                   | Torch       |
| resnet50-imagenet-torch                                   | Torch       |
| resnext101-32x8d-imagenet-torch                           | Torch       |
| resnext50-32x4d-imagenet-torch                            | Torch       |
| vgg11-bn-imagenet-torch                                   | Torch       |
| vgg11-imagenet-torch                                      | Torch       |
| vgg13-bn-imagenet-torch                                   | Torch       |
| vgg13-imagenet-torch                                      | Torch       |
| vgg16-bn-imagenet-torch                                   | Torch       |
| vgg16-imagenet-tf1                                        | TF1         |
| vgg16-imagenet-torch                                      | Torch       |
| vgg19-bn-imagenet-torch                                   | Torch       |
| vgg19-imagenet-torch                                      | Torch       |
| wide-resnet101-2-imagenet-torch                           | Torch       |
| wide-resnet50-2-imagenet-torch                            | Torch       |
| zero-shot-classification-transformer-torch                | Torch       |
| zero-shot-detection-transformer-torch                     | Torch       |

