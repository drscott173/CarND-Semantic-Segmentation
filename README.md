# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder

## Visualization

We used a learning rate of 0.000005 and a dropout rate of 20%, training on 200 epochs with a batch size of 4. We
trained on Google Cloud with a single NVidia P100 GPU.  We first tried training on a personal Mac, which was
entertaining, turning my Mac into a toaster and frying pan due to the heat.  I augmented images by randomly
adjusting contrast, brightness, then shifting the image and result in X and Y, then finally overlaying a
synthetic shadow.  This gave us nearly an infinite training set.

Hear are some decent results:

![Alt text](newest/um_000006.png)
![Alt text](newest/um_000010.png)
![Alt text](newest/um_000014.png)
![Alt text](newest/um_000015.png)
![Alt text](newest/um_000017.png)
![Alt text](newest/um_000032.png)
![Alt text](newest/um_000036.png)

Here are areas we need work:

![Alt text](newest/um_000061.png)
![Alt text](newest/um_000086.png)
![Alt text](newest/umm_000000.png)
![Alt text](newest/umm_000003.png)