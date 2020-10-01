import cv2
import os
import numpy as np

class SimpleDatasetLoader:
    
    def __init__(self, preprocessors=None):
        
        # preprocessors must be a list of preprocessors to be applied
        # if not, then it is initialized as an empty list
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        # imagePaths is a list specifying the file paths to the images in our dataset residing on disk.
        # We will make use of imutils.paths.list_images() function to create imagePaths variable
        # This function returns a list containing the paths of all the images in the
        # parent as well as sub-directories
        #
        # We make the assumption that our datasets are organized on disk 
        # according to the following directory structure:
        # /dataset_name/class/image.jpg
        #
        # here, we use imagePaths = list(imutils.paths.list_images('/dataset_name'))
        # the above line will create flattened list of all the image paths in 
        # dataset_name directory and sub-directories
        
        for (i, imagePath) in enumerate(imagePaths):
            
            image = cv2.imread(imagePath)
            # remember the directory structure, hence we get class(or label) as the second last string
            # in file path separated by operating system's path separator(for windows, it is '\\')
            # /dataset_name/class/image.jpg
            label = imagePath.split(sep=os.path.sep)[-2]
            
            if self.preprocessors is not None:
                # loop over the list of preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))
        
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))


        