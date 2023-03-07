## YOLO2COCO

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://github.com/RapidAI/YOLO2COCO/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/YOLO2COCO?color=9ea"></a>
    <a href="https://github.com/RapidAI/YOLO2COCO/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/YOLO2COCO?color=ccf"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-dfd.svg"></a>
</p>


#### labelImg标注yolo格式数据 → YOLOV5格式
<details>

  - 将[labelImg](https://github.com/tzutalin/labelImg)库标注的yolo数据格式一键转换为YOLOV5格式数据
  - labelImg标注数据目录结构如下（详情参见`dataset/labelImg_dataset`）：
    ```text
      labelImg_dataset
      ├── classes.txt
      ├── images(13).jpg
      ├── images(13).txt
      ├── images(3).jpg
      ├── images(3).txt
      ├── images4.jpg
      ├── images4.txt
      ├── images5.jpg
      ├── images5.txt
      ├── images6.jpg  # 注意这个是没有标注的
      ├── images7.jpg
      └── images7.txt
    ```
  - 转换
    ```shell
    python labelImg_2_yolov5.py --src_dir dataset/labelImg_dataset \
                                --out_dir dataset/labelImg_dataset_output \
                                --val_ratio 0.2 \
                                --have_test true \
                                --test_ratio 0.2
    ```
    - `--src_dir`：labelImg标注后所在目录
    - `--out_dir`： 转换之后的数据存放位置
    - `--val_ratio`：生成验证集占整个数据的比例，默认是`0.2`
    - `--have_test`：是否生成test部分数据，默认是`True`
    - `--test_ratio`：test数据整个数据百分比，默认是`0.2`

  - 转换后目录结构（详情参见`dataset/labelImg_dataset_output`）：
    ```text
    labelImg_dataset_output/
      ├── classes.txt
      ├── images
      │   ├── images(13).jpg
      │   ├── images(3).jpg
      │   ├── images4.jpg
      │   ├── images5.jpg
      │   └── images7.jpg
      ├── labels
      │   ├── images(13).txt
      │   ├── images(3).txt
      │   ├── images4.txt
      │   ├── images5.txt
      │   └── images7.txt
      ├── non_labels        # 这是没有标注图像的目录，自行决定如何处置
      │   └── images6.jpg
      ├── test.txt
      ├── train.txt
      └── val.txt
    ```
  - 可以进一步直接对`dataset/labelImg_dataset_output`目录作转COCO的转换
    ```shell
    python yolov5_2_coco.py --dir_path dataset/lablelImg_dataset_output
    ```

</details>

#### YOLOV5格式数据 → COCO
<details>

  - 可以将一些背景图像加入到训练中，具体做法是：直接将背景图像放入`backgroud_images`目录即可。
  - 转换程序会自动扫描该目录，添加到训练集中，可以无缝集成后续[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)的训练。
  - YOLOV5训练格式目录结构（详情参见`dataset/YOLOV5`）：
      ```text
      YOLOV5
      ├── classes.txt
      ├── background_images  # 一般是和要检测的对象容易混淆的图像
      │   └── bg1.jpeg
      ├── images
      │   ├── images(13).jpg
      │   └── images(3).jpg
      ├── labels
      │   ├── images(13).txt
      │   └── images(3).txt
      ├── train.txt
      └── val.txt
      ```

  - 转换
      ```shell
    python yolov5_2_coco.py --dir_path dataset/YOLOV5 --mode_list train,val
    ```
    - `--dir_path`：整理好的数据集所在目录
    - `--mode_list`：指定生成的json，前提是要有对应的txt文件，可单独指定。（e.g. `train,val,test`）

  - 转换后目录结构（详情参见`dataset/YOLOV5_COCO_format`）：
  ```text
  YOLOV5_COCO_format
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  ├── train2017
  │   ├── 000000000001.jpg
  │   └── 000000000002.jpg  # 这个是背景图像
  └── val2017
      └── 000000000001.jpg
  ```
</details>

#### YOLOV5 YAML描述文件 → COCO
<details>

  - YOLOV5 yaml 数据文件目录结构如下（详情参见`dataset/YOLOV5_yaml`）：
      ```text
      YOLOV5_yaml
      ├── images
      │   ├── train
      │   │   ├── images(13).jpg
      │   │   └── images(3).jpg
      │   └── val
      │       ├── images(13).jpg
      │       └── images(3).jpg
      ├── labels
      │   ├── train
      │   │   ├── images(13).txt
      │   │   └── images(3).txt
      │   └── val
      │       ├── images(13).txt
      │       └── images(3).txt
      └── sample.yaml
      ```

  - 转换
    ```shell
    python yolov5_yaml_2_coco.py --yaml_path dataset/YOLOV5_yaml/sample.yaml
    ```

  #### darknet格式数据 → COCO
  - darknet训练数据目录结构（详情参见`dataset/darknet`）：
    ```text
    darknet
    ├── class.names
    ├── gen_config.data
    ├── gen_train.txt
    ├── gen_valid.txt
    └── images
        ├── train
        └── valid
    ```

  - 转换
    ```shell
    python darknet2coco.py --data_path dataset/darknet/gen_config.data
    ```
</details>

#### 可视化COCO格式下图像
<details>

```shell
python coco_visual.py --vis_num 1 \
                    --json_path dataset/YOLOV5_COCO_format/annotations/instances_train2017.json \
                    --img_dir dataset/YOLOV5_COCO_format/train2017
```

- `--vis_num`：指定要查看的图像索引
- `--json_path`：查看图像的json文件路径
- `--img_dir`: 查看图像所在的目录

</details>


#### 相关资料
- [MSCOCO数据标注详解](https://blog.csdn.net/wc781708249/article/details/79603522)
