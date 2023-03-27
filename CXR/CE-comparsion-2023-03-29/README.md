# BinaryCrossEntropy a survey 2023-03-29

System hardware:

- System: Ubuntu 20.04 
- CPU count:	14
- GPU count:	1
- GPU type :	NVIDIA GeForce RTX 3050 Ti Laptop GPU


In this experimental, we survey the BCE loss on varitely model architecture, including densenet121 (applied on CXR mostly),
swin, coat, beit, convext...

## Configuration

The origin dataset are split into training, validation and testing, the support list in the following:

| Disease/DataSet        | Train | Valid | Test |
|------------------|--------|---------|---------|
| 主動脈硬化(鈣化)   | 67     | 8       | 9       |
| 動脈彎曲         | 74     | 9       | 9       |
| 肺野異常         | 24     | 3       | 3       |
| 肺紋增加         | 106    | 14      | 13      |
| 脊椎病變         | 114    | 15      | 14      |
| 心臟肥大         | 33     | 4       | 4       |
| 肺尖肋膜增厚      | 29     | 4       | 3       |

The basis configuraions are the following:

```python
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('--base_root', type=str, default='/home/dongdong/Medical-Image-Analysis/CXR/xray_jpg/', help='path to the base data directory')
    parser.add_argument('--csv_name', type=str, default='chest_dongdong.csv', help='name of the CSV file containing the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--save_name', type=str, default='model.pth', help='name of the file to save the trained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for the optimizer')
    parser.add_argument('--pretrained', type=bool, default=True, help='transfer learning or not')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    args = parser.parse_args()
    return args
```

## Experimental Results

For densenet, we train 5 epochs without freeze backbone, however, in transformer architecture, we found that it is hard to converge when jointly trianing, therefore, we first train 10 epochs with freeze backbone then 10 epochs with all parameters.

The training result are shown below:

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

### Related architecture

- DenseNet
   - original source: https://github.com/baaivision/EVA
   - paper: https://arxiv.org/pdf/1608.06993.pdf
- BEiT
   - paper: https://arxiv.org/pdf/2211.07636.pdf
- MaxViT
   - paper: https://arxiv.org/pdf/2204.01697.pdf
- CoATNet
   - paper: https://arxiv.org/pdf/2106.04803.pdf

<!-- <style>
    .max {
        color: red;
        font-weight: bold;
    }
</style> -->

## Future work

- [ ] (Another loss function)
- [ ] (10 times average performance)
- [ ] (more training strategy)
