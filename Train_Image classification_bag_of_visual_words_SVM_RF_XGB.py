"""
The following code worked on Python 3.5.6; openCV 3.1.0; Spyder 3.3.1; sklearn 0.20.0; joblib 0.12.5 - created a custom environment of Anaconda - might not be possible on all computers.
All section images resized to 1000 x 1000.
Images used for test are completely different that the ones used for training.
744 and 716 test images for MVM- and MVM+, respectively.
7036 and 6195 train images for MVM- and MVM+, respectively (augmented).
"""


import cv2
import numpy as np
import os


# Get the training classes names and store them in a list
#Here we use folder names for class names

#train_path
train_path = 'SVM/train_DA_70'  # Folder Names are MVM- and MVM+
training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory define a function
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
    
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Create feature extraction and keypoint detector objects
#SIFT is not available anymore in openCV (patented- need to get a license?)   
# Create List where all the descriptors will be stored
des_list = []

#BRISK is a good replacement to SIFT (patented - license needed). ORB also works - may need to try for this dataset.
#brisk = cv2.BRISK_create(250) #try 250 and 500 features
#orb = cv2.ORB_create(250) #try 250 and 500 features
sift = cv2.xfeatures2d.SIFT_create(500) #try 250 and 500 features

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = sift.detectAndCompute(im, None) 
    des_list.append((image_path, des))
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

#kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)  

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 300  #try kmeans with 100, 200 and (maybe 300) clusters
voc, variance = kmeans(descriptors_float, k, 1)

# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

#Train an algorithm to discriminate vectors corresponding to positive and negative training images

#Train the Linear SVM
#from sklearn.svm import LinearSVC
#clf = LinearSVC(max_iter=10000)  #Default of 100 is not converging
#clf.fit(im_features, np.array(image_classes))

#Train Random forest to compare how it does against SVM
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 200, random_state=42) #try 100 and 200 estimators
#clf.fit(im_features, np.array(image_classes))

#Train the xgboost to compare how it does against SVM and RF
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators = 200) #try n_estimators = 200; the default is 100
clf.fit(im_features, np.array(image_classes))


# Save the SVM or RF, change pkl file name accordingly
#Joblib dumps Python object into one file
from sklearn.externals import joblib
joblib.dump((clf, training_names, stdSlr, k, voc), "PE_DA_bovw_sift500_km300_XGB200.pkl", compress=3)
