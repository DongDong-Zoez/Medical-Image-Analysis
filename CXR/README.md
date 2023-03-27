# Chest Xray MultiLabel Classification

In this project, we are collaborating with the E-Da Hospital to analyze patients' chest imaging for a multi-label deep learning image classification task. Our goal is to achieve the state-of-the-art (SOTA) performance on the dataset. More details can be found in the Meetings folder.

The data contain 1635 patients CXR and mostly are normal one.

## Support Set

The origin data have 19 classes, we simply catogery it to 8 classes (with normal), the support of the data are the following:

| Disease/DataSet        | Train | Valid | Test |
|------------------|--------|---------|---------|
| 主動脈硬化(鈣化)   | 67     | 8       | 9       |
| 動脈彎曲         | 74     | 9       | 9       |
| 肺野異常         | 24     | 3       | 3       |
| 肺紋增加         | 106    | 14      | 13      |
| 脊椎病變         | 114    | 15      | 14      |
| 心臟肥大         | 33     | 4       | 4       |
| 肺尖肋膜增厚      | 29     | 4       | 3       |

The imbalance ratio of the dataset is 3.9% (positive samples divided by negative samples)

## Current Base Result

Currrently, the result we have study 

| ModelName | Train AUC | Valid AUC | Test AUC | params(M) |
|-----------|-----------|-----------|----------|--------|
|tv_densenet121|0.8409|0.8176|0.8599|7.98|
|vit_base_patch32_224_clip_laion2b|0.6365|0.6515|0.5585|88.2|
|beit_base_patch16_224_in22k|0.7518|0.7257|0.7705|102.56|
|beit_large_patch16_224_in22k|0.7512|0.7458|0.7976|<a class="max">325.79</a>|
|maxvit_tiny_224|0.8680|0.8171|0.8425|30.92|
|maxvit_small_224|0.9044|0.8259|0.8382|68.93|
|maxvit_base_224|0.8634|0.8421|0.8443|119.47|
|maxvit_large_224|0.8745|0.8365|0.8366|211.79|
|coatnet_0_rw_224|0.9063|0.8330|<a class="max">0.8784</a>|27.44|
|coatnet_1_rw_224|<a class="max">0.9249</a>|<a class="max">0.8500</a>|0.8665|41.72|
|coatnet_2_rw_224|0.8721|0.8186|0.8337|73.89|
|coatnet_3_rw_224|0.8445|0.7990|0.8266|165.17|