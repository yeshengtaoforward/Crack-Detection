# Deep supervised pavement crack segmentation algorithm based on multi-scale attention and feature fusion 

## Abstract
Aiming at the problem of incomplete crack segmentation and missed detection in 
pavement crack detection under complex background and unbalanced crack pixel ratio, this paper 
proposes a deep supervised pavement crack segmentation algorithm with multi-scale attention and 
feature fusion. Firstly, the EfficientNet-B3 is improved by using the ECA (efficient channel attention) 
attention mechanism and used as the encoding part of the network model to accelerate the attention to 
crack pixels. Secondly, a multi-scale channel attention module (MSCA) is designed, which uses the 
cascade parallel strategy of dilated convolution to extract key contextual information and enhance the 
perception of small cracks. Finally, multiple side feature maps are integrated in the auxiliary network 
in a feature pyramid manner, and a deep supervision mechanism is introduced to accelerate the 
convergence of the model and improve the effect of crack detection. Experiments are carried out on 
the CRACK500, CFD, and DeepCrack datasets. The F1 in the detection results can reach 75.87%, 
66.80%, and 86.46% respectively, which is better than the current advanced crack segmentation 
methods and has certain application value. 

<hr>

## Train the model
```commandline
python train.py --path_imgs "path_to_image_folder" --path_masks "path_to_mask_folder" --out_path "out_path_to_store_best_model_and_logs"
```

<hr>

## Evaluate Model
```commandline
python evaluate.py --path_imgs "path_to_image_folder" --path_masks "path_to_mask_folder" --model_path "path_to_saved_model" --result_path "path_to_save_results_from_test" --plot_path "path_to_store_plots"
```
<hr>

## Model Architecture
<br>
<figure>
<img src="assets/MainArchitecture.drawio Conf.png">
<figcaption align = "center"><b>Fig.1 - Main Architecture</b></figcaption>
</figure>
<br>
<br>
<figure>
<img src="assets/MultiScaleAttention.drawio.png">
<figcaption align = "center"><b>Fig.2 - Multi Scale Attention</b></figcaption>
</figure>
<hr>

## Dataset
Crack500 Dataset: This dataset includes 500 images of pavement cracks with
a resolution of 2000 x 1500 pixels collected at Temple University campus using a
smartphone by [1]. Each image is annotated on a pixel level. Images this large
won’t fit on GPU memory; therefore, [1] patched each image into 16 smaller
images. The final resolution of each image was 640x320, and the dataset has 1896
images in training set, 348 images in validation set, and 1124 images in testing
set. The comparisons with state-of-the-art models were made on the results from
the testing set

<hr>

## Results

<table>
    <th>Dataset</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
    <th>IoU</th>
    <tr>
    <td>Crack500</td>
    <td>97.4</td>
    <td>0.763</td>
    <td>0.790</td>
    <td>0.775</td>
    <td>0.621</td>
    </tr>
</table>
<br>
<figure>
<figcaption><b>RGB Image</b>  &ensp;&ensp;&ensp;&ensp;  <b>Ground Truth</b>&ensp;&ensp;&ensp;&ensp;<b>Prediction (Model Output)</b></figcaption>

<img src="assets/result_imgs_crack500.drawio.png">
<figcaption align="center"><b>Fig 3: Result From model</b></figcaption>
</figure>
<hr>

<hr>

## ToDo
- Dockerise code

## References
[1] F. Yang, L. Zhang, S. Yu, D. Prokhorov, X. Mei, and H. Ling, “Feature pyramid and
hierarchical boosting network for pavement crack detection,” IEEE Transactions
on Intelligent Transportation Systems, vol. 21, no. 4, pp. 1525–1535, 2020.

[2] Lyu, C., Hu, G. & Wang, D. Attention to fine-grained information: hierarchical multi-scale network for retinal vessel segmentation. Vis Comput 38, 345–355 (2022). https://doi.org/10.1007/s00371-020-02018-w
