# pytorch
guided feature inversion: @article{du2018towards,
                            title={Towards Explanation of DNN-based Prediction with Guided Feature Inversion},
                            author={Du, Mengnan and Liu, Ninghao and Song, Qingquan and Hu, Xia},
                            journal={arXiv preprint arXiv:1804.00506},
                            year={2018}
                          }


this is an implementation of the algorithm developed in the above paper, Towards Explanation of DNN-based Prediction with Guided Feature Inversion. This algorithm has two steps. the first step is to find the saliency map for the input image by guided feature inversion, and the second step is to find the class discriminative mask for the target object. 

in order to use the first algorithm, you just need to install the required packages, mentioned in the requirements file, and specify the directory of the image you want to run the algorithm for it. There are some example images in this repository that you can utilize them. however, before using the algorithms you just need to adjust the images size to (224 * 224). the outputs are a heat-map of the saliency, a regular mask, and a background mask for the input image, which will be saved in the generated file directory after running the first algorithm.  


The second step will find the class discriminative mask for the target object from an input image. it will use the achieved weights from the first algorithm to initialize the W weights and it tries to change the weights so that the activation unit for the target object increase.

to use the second algorithm, you need to specify image directory as well as the target class of the input image, since VGG19 is used and VGG19 is trained over ImageNet dataset, you can find the target class for each image from [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). the output of the second algorithm will be a heat-map and a mask for the target class. 

<img src="https://github.com/arminkhayyer/pytorch/blob/armin/11.jpg"> </img> input image(target class elephant: 386, zebra: 340)

<h1 style="color:red;">first algorithm results </h1>

<img src="https://github.com/arminkhayyer/pytorch/blob/armin/generated/Inv_Image_Layer_11.jpg"> </img> output saliency image

<img src="https://github.com/arminkhayyer/pytorch/blob/armin/generated/heatmap_11.png"> </img> output heat map 

<h1 style="color:red;">second algorithm results </h1>

<img src="https://github.com/arminkhayyer/pytorch/blob/armin/generated/Inv_Image_Layer_11.jpg"> </img> output saliency image

<img src="https://github.com/arminkhayyer/pytorch/blob/armin/generated/heatmap_11.png"> </img> output heat map 




