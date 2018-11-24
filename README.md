# pytorch
guided feature inversion: @article{du2018towards,
                            title={Towards Explanation of DNN-based Prediction with Guided Feature Inversion},
                            author={Du, Mengnan and Liu, Ninghao and Song, Qingquan and Hu, Xia},
                            journal={arXiv preprint arXiv:1804.00506},
                            year={2018}
                          }

this is an implemention of the algorithm developed in the above paper, Towards Explanation of DNN-based Prediction with Guided Feature Inversion. 
this algorithm has two steps. the first step is to find the saliency map for the input image by guided feature inversion, and the second step 
is to find the class discriminative mask for the target object. 

in order to use the first algorithm you just need to install the required packages, and specify the directory of the image you want to run the algorithm for it.
there are some example images in the repository that you can utilize them. however, before using the algorithms you just need to adjust the images size to (224 * 224)

the outputs are a heatmap of the saliency, a regular mask, and a backgraoung mask for the input image. 
