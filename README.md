## Meta Learning Algos

### Implementation of OpenAI Reptile algorithm with PyTorch.

#### In order to run this algorithm, first create a folder and rename it as data. Next, copy over images_background.zip and images_evaluation.zip from https://github.com/brendenlake/omniglot/tree/master/python and place it inside the folder, data. Unzip both the zip files and you should see two folders should be created name images_background and images_evaluation. 

#### Next to run an experiment, please go to reptileExperimentOmniglot.py, and at the bottom you can specify how many classes we (N) and how many examples (K) we will be using for training. Currently doing a 5 way 5 shot classification with 100K total meta iterations, training takes roughly 12 hours to complete. 

#### The hyperparameters for this experiment are: learning rate of learner = 1e-3, meta learner learning rate (lr_outer) = 1, N = 5, K = 5 (5 way 5 shot learning). DATA_PATH = current working directory (Make sure data folder is also in your current working directory). I use SGD as the optimizer for taking k gradient steps and Adam optimizer for the meta learner. Furthermore, specify the name of the directory where you want to save checkpoints during training. If the directory does not exist, constructor in reptileExperimentOmniglot.py will create the directory. 

#### Once the experiment is done running, you can create an python notebook or load the experiment (exp.loadState()) stored in the directory specified and then call (exp.evaluateModelTest()) to see how well Reptile is able to adapt to learning new characters. This method randomly samples a character from the testing set which is specified in the data loader, we take K example images for this character and make it a our training step and let the rest be part of the testing set. We perform fine tuning by doing 50 iterations of SGD as specified by the paper and then evaluate on the testing set. When doing this, I have observed that Reptile has gotten 100% accuracy every time. This indicates that the weight initializations that was found during meta training has allowed our model to easily adapt to learning other characters. 

#### models.py contains the definition for the 4 layer convolutional neural network with batch normalization as specified in the paper. 


#### dataloader.py contains functions that will be used to manipulate images found in data/images_evaluation and data/images_background, this file is mainly used to create mini batches and augmentation of training set for training our meta learner. 

### ReptileSineWaveExperiment.ipynb is a python notebook that demonstrates how well Reptile performs on a simple sine regression task.


### The link to the original paper is: https://arxiv.org/pdf/1803.02999.pdf. 


