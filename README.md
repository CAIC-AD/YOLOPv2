

<div align="center">
<h1> YOLOPv2: Better, Faster, Stronger for Panoptic driving Perception </h1>
<!-- <--!span><font size="5", > Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer
</font></span> -->

  Cheng Han*, Qichao Zhao, Shuyi Zhang,   <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN">Jinwei Yuan</a>, Zhenlin Zhang
 
<div><a href="https://arxiv.org/pdf/2206.04584.pdf">[arXiv Preprint]</a></div> 

</div>

## News

* `August 24, 2022`: We've released the tech report for **YOLOPv2**. This work is still in progress and code/models are coming soon. Please stay tuned! ☕️

## Introduction

We present an excellent multi-task network based on YOLOP,which is called **YOLOPv2: Better, Faster, Stronger for Panoptic driving Perception**.
The advantages of **YOLOPv2** can be summaried as below:
* **Better**: we proposed the end-to-end perception network which possess better feature extraction backbone, better bag-of-freebies were developed for dealing with the training process.
* **Faster**: we employed more efficient ELAN structures to achieve reasonable memory allocation for our  model. 
* **Stronger**: the proposed model possess more stable and reasonable structure has powerful robustness for adapting to various scenarios.
## Results
We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.

#### Model parameter and inference speed
|        Model       |    Size     |   Params   |  Speed (fps) |
|:------------------:|:-----------:|:----------:|:-----------:|
|       `YOLOP`      |     640     |    7.9M    |     49      |
|     `HybridNets`   |     640     |    12.8M   |     28      |
|    **`YOLOPv2`**   |     640     |  **38.9M** |   **91 (+42)**    |


#### Traffic Object Detection Result
|        Model       |     mAP@0.5 (%)   |Recall (%)  |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |        60.2      |   81.3     |  
|      `DLT-Net`     |        68.4      |  89.4     |
|   `Faster R-CNN`   |          55.6      | 77.2     |
|      `YOLOv5s`     |         77.2      | 86.8     |
|       `YOLOP`      |        76.5      | 89.2     |   
|     `HybridNets`   |          77.3      | **92.8**   | 
|    **`YOLOPv2`**   |       **83.4(+6.1)**    |   91.1(-1.7)     |

### Drivable Area Segmentation

|       Model      | Drivable mIoU (%) |
|:----------------:|:-----------------:|
|    `MultiNet`    |        71.6       |
|     `DLT-Net`    |        71.3       |
|     `PSPNet`     |        89.6       |
|      `YOLOP`     |        91.5       |
|     `HybridNets` |        90.5       |
|     `YOLOPv2`    |   **93.2(+1.7)**  |

### Lane Line Detection
 
|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |        26.2       |
|   `HybridNets`   |     85.4     |        **31.6**     |
|    **`YOLOPv2`** |   **87.3(+1.9)**   |      27.2(-4.4)  |

## Visualization






## Models

coming soon.

<!-- ## Usage

coming soon. -->


## License

GKT is released under the [MIT Licence](LICENSE).

<!-- ## Citation

If you find YOLOPv2 is useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.

```bibtex
@article{GeokernelTransformer,
  title={Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer},
  author={Chen, Shaoyu and Cheng, Tianheng and Wang, Xinggang and Meng, Wenming and Zhang, Qian and Liu, Wenyu},
  journal={arXiv preprint arXiv:2206.04584},
  year={2022}
}
``` -->
