import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.autograd import Variable

#this file contains the data loader for omniglot 
#this takes in the current working directory and it must have a data folder where both 
#the images_background and images_evaluation are located. You also specify what sort of 
#task will we doing i.e K shot N way classification. 
class OmniglotDataLoader():
    def __init__(self, root_dir, K, N, image_size = (28,28), trainSize = 1200, minibatch_size = 10):
        self.root_dir = root_dir 
        if not os.path.exists(self.root_dir):
            print('inputted path does not exist, please give valid path')
            return 
        subdir1 = '/data/images_background'
        subdir2 = '/data/images_evaluation'
        languageDir1 = self.root_dir + subdir1 
        languageDir2 = self.root_dir + subdir2
        if not os.path.exists(languageDir1) or not os.path.exists(languageDir2):
            print('data directory does not exist or have not unzipped images_background.zip or images_evaluate.zip. Can download zip from: https://github.com/brendenlake/omniglot/tree/master/python')
        print('file path exists with Omniglot dataset ')
        self.image_size = image_size
        self.characters = self.getAllCharactersFrom(languageDir1, languageDir2)
        self.training_set  = self.characters[:trainSize]
        self.test_set = self.characters[trainSize:]
        
        self.numClasses = N 
        self.numShots = K
        self.minibatch_size = minibatch_size 
        
    #In this method, it aggregates all the characters across both the images_background 
    #directory and images_evaluation directory. This will return a list of paths to all characters 
    #in both directory. In total should have 1623 characters. 
    def getAllCharactersFrom(self, languageDir1, languageDir2):
        #get all characters from languageDir1 
        alphabets_in_languageDir1 = os.listdir(languageDir1)
        alphabets_in_languageDir2 = os.listdir(languageDir2)
        #tuple of character and character path 
        characters_in_both_dirs = []
        
        #get all character image paths
        counter = 0 
        for alphabet in alphabets_in_languageDir1:
            #image_background/alphabet/<ALL CHARACTERS>
            character_in_alphabet = os.listdir(languageDir1 + '/' + alphabet)
            for character in character_in_alphabet:
                character_path = languageDir1 + '/' + alphabet + '/' + character
                character_name = 'character' + str(counter)
                characters_in_both_dirs.append((character_name,character_path))
                counter+=1
                
        for alphabet in alphabets_in_languageDir2:
            #image_background/alphabet/<ALL CHARACTERS>
            character_in_alphabet = os.listdir(languageDir2 + '/' + alphabet)
            for character in character_in_alphabet:
                character_path = languageDir2 + '/' + alphabet + '/' + character
                character_name = 'character' + str(counter)
                characters_in_both_dirs.append((character_name,character_path))
                counter+=1                        
        return characters_in_both_dirs
            
    #get training and val set by randomly choosing N classes 
    #and from N classes get K + 1 examples, where the K + 1th example
    #is part of the validation set 
    def get_data_set(self, class_names):
        training_set_img_paths = []
        validation_set_img_paths = []
        testing_set_img_paths = []
        for k in class_names: 
            char_path = self.meta_dataset_dir + '/' + k
            image_names = os.listdir(char_path)
            image_names = [char_path + '/' + x for x in image_names]
            random.shuffle(image_names)
            training_set_img_paths.append((image_names[:self.num_shots], self.class_to_idx[k]))
            validation_set_img_paths.append((image_names[self.num_shots:self.num_shots+1], self.class_to_idx[k]))
            testing_set_img_paths.append((image_names[self.num_shots+1:], self.class_to_idx[k]))
            
        training_set = self.augment_and_create_dataset(training_set_img_paths)
        validation_set = self.augment_and_create_dataset(validation_set_img_paths)
        testing_set = self.augment_and_create_dataset(testing_set_img_paths)
                            
        return training_set, validation_set, testing_set
            
        
    #given a list of training paths and labels, validation paths and labels, 
    #this method first opens the image and then converts the image to a tensor. 
    #Did not normalize since the original Reptile paper made no mention to normalize 
    #the images. Furthermore, if the image is in the training set, I augment the training 
    #set by rotating the image by 90, 180, and 270 degrees. The returned output returns 
    #the tensored training set and validation set. 
    def augment_and_create_dataset(self, training_paths, training_labels, validation_paths,
                                  validation_labels):
        transform = tv.transforms.Compose([tv.transforms.Resize(self.image_size),
                                           tv.transforms.ToTensor()])
        training_set = []
        validation_set = []
        angle_variants = [90, 180, 270]
        for idx,image_path in enumerate(training_paths): 
            img = Image.open(image_path).convert("RGB")
            transformed_img = transform(img)
            training_set.append((transformed_img, training_labels[idx]))
            for angle in angle_variants: 
                #augment the training_set by including rotating by 90, 180, 270 
                imgRotate = img.rotate(angle)
                transformed_img = transform(imgRotate)
                training_set.append((transformed_img, training_labels[idx]))
        
        for idx, image_path in enumerate(validation_paths):
            img = Image.open(image_path).convert("RGB")
            transformed_img = transform(img)
            validation_set.append((transformed_img, validation_labels))
        
        return training_set, validation_set
    
    #based on the training set and labels, this method creates a set of mini batches 
    #based on what was specified in the constructor. This method shuffles the training set and 
    #after shuffling creates a mini batch. 
    def create_mini_batches(self, training_tensorList_and_labels, validation_tensorList_and_labels):
        #separate training set into minibatches and return 
        random.shuffle(validation_tensorList_and_labels)
        #(image tensors (batch_size * 3 * 28 * 28), labels (batch_size * 1))
        training_batch = []
        validation_batch = []
        training_indices = list(range(len(training_tensorList_and_labels)))
        random.shuffle(training_indices)
        training_tensors = [tensor for (tensor, label) in training_tensorList_and_labels]
        training_labels = [label for (tensor, label) in training_tensorList_and_labels]
        validation_tensor = [tensor for (tensor, label) in validation_tensorList_and_labels]
        validation_labels = [label for (tensor, label) in validation_tensorList_and_labels]
        validation_tensor_batch = torch.stack(validation_tensor, dim=0)
        validation_label_batch = torch.Tensor(validation_labels).long()
        validation_batch.append((validation_tensor_batch,validation_label_batch))
        #number of mini batches dependent on how many training samples 

        
        num_mini_batches = int(len(training_tensors)/self.minibatch_size)
        
        for mini_batch in range(num_mini_batches):
            chosen_idx = training_indices[mini_batch*self.minibatch_size:(mini_batch+1)*self.minibatch_size]
            chosen_tensors = [training_tensors[i] for i in chosen_idx]
            chosen_labels  = [training_labels[i] for i in chosen_idx]
            curr_training_batch_t = torch.stack(chosen_tensors, dim=0)
            curr_training_batch_l = torch.Tensor(chosen_labels).long()
            training_batch.append((curr_training_batch_t,curr_training_batch_l))
        return training_batch, validation_batch 
        
    
    #This method is used for sampling tasks from the training set. This 
    #method chooses which N tasks to sample. From those N tasks, 
    #we then choose K examples from each. After this we then 
    #create mini batches 
    def sample_task_from_training_and_val(self):
        #choose a random class to classify from training set 
        chosen_dataset = self.training_set
        random.shuffle(chosen_dataset)
        chosen_classes = chosen_dataset[:self.numClasses]
        #recall that chosen sample contain paths to the images(.png files)
        training_set = []
        validation_set = []
        #choose K examples from each chosen_class 
        labels = list(range(5))
        idx = 0 
        
        #get paths to all images
        for charName, charPath in chosen_classes:
            #get all the image files in directory 
            images = os.listdir(charPath)
            random.shuffle(images)
            training_images = images[:self.numShots]
            training_images = [charPath + '/' + i for i in training_images]
            currLabel = [labels[idx]]*len(training_images)
            training_set.append([(image, label) for (image, label) in zip(training_images, currLabel)])
            validation_image = images[self.numShots:self.numShots+1]
            validation_images = [charPath + '/' + i for i in validation_image]
            validation_set.append((validation_images,labels[idx]))
            idx+=1

        training_tensorList_and_labels = []
        validation_tensorList_and_labels = []
        
        #get transform and augment training images
        for idx in range(self.numClasses): 
            currTrainingPaths =  [image_path for (image_path, image_label) in training_set[idx]]
            currTrainingLabels = [image_label for (image_path, image_label) in training_set[idx]]
            currValidationPaths = validation_set[idx][0]
            currValidationLabels = validation_set[idx][1]
            curr_training_tensors_and_labels, curr_val_tensors_and_labels = self.augment_and_create_dataset(currTrainingPaths, currTrainingLabels, currValidationPaths, currValidationLabels)
            training_tensorList_and_labels = training_tensorList_and_labels + curr_training_tensors_and_labels
            validation_tensorList_and_labels = validation_tensorList_and_labels + curr_val_tensors_and_labels
            
        return self.create_mini_batches(training_tensorList_and_labels,validation_tensorList_and_labels )
        
         