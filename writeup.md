## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_GTI_Right_image0333.png
[image2]: ./output_images/hog_GTI_image1007.png
[image3]: ./output_images/normalize_feats.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell of the IPython notebook under the title `1.1 Histogram of Oriented Gradient (HOG)` and the `get_hog_features()` function is in lines 7 through 24 of the file called `lesson_functions.py`.  
I explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to the `hog()` function from `skimage.feature`.  I select test images from each of the two classes of images with or without a vehicle and displayed three layers of each image as well as the HOG features for each layer.

Here are examples using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`.

For an image that contains a vehicle:
![alt text][image1]

For an image that doesn't contain any vehicle:
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that `orientations = 11` gives enough resolution of orientation angle binning and not too much. I settled on `pixels_per_cell = (16, 16)` because 16x16 is a good cell size for the gradient histogram to have a significant number of pixels to sum up and have a smooth result without too much noise. I didn't change the default number of `cells_per_block = 2` which means to normalize the histogram over blocks of 2x2 cells. I was able to achieve good results without varying this parameter.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First, I prepare features and labels for training a SVM classifier. I read in the images with and without vehicles in the code cell under the title `1.3 Feature Normalization`. I then use function `extract_features()` which is in line 53 to 71 in the `lesson_functions.py` to extract binned color features, color histogram features as well as HOG features from each image. The length of a feature vector is 4,356. To normalize features to have mean zero and unit standard deviation, I apply `StandardScaler()` from `sklearn` to the vertically stacked feature array. An example of feature normalization is:

![alt text][image3]


Next, I prepare labels by creating an array with `1`'s for feature row numbers with vehicles and `0`'s for feature row numbers without any vehicle. I then use `train_test_split()` to randomly splits the data into 80% for training model and 20% for testing the model. This is done in the code cells under the title `1.4 Preparation`.

Finally, I train a linear SVM model by using cross validation to search for the optimal `C` parameter using the `GridSearchCV()` function. In the code cells under the title `1.5 Tune SVM parameters`, I found that `C = 0.0003` is the best for this set of data. Then I train a SVM model using this parameter and test it on the testing data to confirm that the model (svc_final) works well on unseen data: `Accuracy = 0.9958` and `F1_score = 0.9957`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

