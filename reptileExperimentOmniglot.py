import os
import time
import torch
from torch import nn
from dataloader import *
from models import *
from torch.autograd import Variable

class ReptileExperiment(): 
    def __init__(self, model, lr_inner, lr_outer, 
                 dataLoader, meta_iter, K, checkpoint_dir,
                 epsilon = 1,
                 cross_entropy_loss= nn.CrossEntropyLoss()):
        self.model = model 
        self.lr_inner = lr_inner 
        self.lr_outer = lr_outer 
        self.epsilon = epsilon
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_outer)
        self.loss_func = cross_entropy_loss
        self.dataLoader = dataLoader 
        self.meta_iter = meta_iter 
        self.totalGradSteps = K 
        self.checkpoint_dir = checkpoint_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.training_loss = []
        self.validation_loss = []
        self.curr_iter = 0
        self.checkpoint_path = os.path.join(checkpoint_dir,"checkpoint.pth.tar")
        self.config_path =  os.path.join(checkpoint_dir,"config.txt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)
        if self.checkpoint_exists():
            self.loadState()
        else:
            self.saveState()
        
    
    #save model state
    #save meta_optimizer state
    #save data loader (contains training, testing split)
    #save totalGradSteps 
    #save total meta iter 
    #save curr iter
    #save training_loss/validation loss
    #save lr_inner/lr_outer
    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'model': self.model.state_dict(),
                'meta_optimizer': self.meta_optimizer.state_dict(),
                'training_loss' : self.training_loss,
                'validation_loss' : self.validation_loss,
                'curr_iter' : self.curr_iter,
                'meta_iter' : self.meta_iter,
                'lr_inner' : self.lr_inner, 
                'dataLoader' : self.dataLoader,
                'totalGradSteps' : self.totalGradSteps
               }
    #this method retrieves saved data from disk and updates 
    #their respective fields. 
    def load_state_dict(self, checkpoint):
        # load from pickled checkpoint
        self.model.load_state_dict(checkpoint['model'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.training_loss = checkpoint['training_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.curr_iter = checkpoint['curr_iter']
        self.meta_iter = checkpoint['meta_iter']
        self.lr_inner = checkpoint['lr_inner']
        self.dataLoader =  checkpoint['dataLoader']
        self.totalGradSteps = checkpoint['totalGradSteps']

        for state in self.meta_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    
    #this method saves all the data we wish to save in the case 
    #that the computer shuts off or if there is any issues
    def saveState(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w+') as f:
            print(self, file=f)
    
    #this method loads the saved data from disk
    def loadState(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    #based on the checkpoint_path specified, this method determines whether we need to 
    #load from disk 
    def checkpoint_exists(self):
        return os.path.exists(self.checkpoint_path) and os.path.exists(self.config_path)
       
        
    #during training/after training, we evaluate the predictive capabilties of the model 
    #we go through the batch of examples given and then compute the accuracy and cross
    #entropy loss. 
    def evaluateModel(self, model, chosen_batch, mode='during_training'):
        model.eval()
        loss = 0 
        accuracy = 0
        total_num_examples = 0
        with torch.no_grad(): 
            for batch, labels in chosen_batch:
                batch_var = Variable(batch.to(self.device))
                label_var = Variable(labels.squeeze().to(self.device))
                label_pred = model(batch_var)
                _,pred_class = torch.max(label_pred, 1)
                total_num_examples+= label_var.shape[0]
                accuracy += (torch.sum((pred_class == label_var).data).item())
                cross_entropy_loss = self.loss_func(label_pred,label_var)
                
                loss += cross_entropy_loss.item()  
        accuracy = accuracy/total_num_examples
        if mode == 'during_training':
            self.model.train()
        return loss, accuracy 
    
    #this method is specifically used once the model is done training. This method 
    #specifically samples from the test set rather than the training set. The reason we 
    #do this is because we want to see how well the model can adapt to learning new characters. 
    #This method chooses 1 class from the testing set and from that 1 class, we choose K examples
    #as a training set and the rest as a testing set. We then perform fine tuning with the K examples 
    #from the training set. Furthermore, when doing fine tuning we take 50 gradient steps rather 
    #than 5 gradient steps as outlined by the paper. We furthermore print the loss and accuracy. 
    #When evaluating the test set with the model, I am observing that it is getting to 100% accuracy. 
    def evaluateModelTest(self):
        #sample a task from test_set 
        #choose K examples from test_set and train on those         
        chosen_dataset = self.dataLoader.test_set 
        random.shuffle(chosen_dataset)
        chosen_class = chosen_dataset[0]
        images = [chosen_class[1] + '/' + image for image in os.listdir(chosen_class[1])]
        training_images = images[:self.dataLoader.numShots]
        test_images = images[self.dataLoader.numShots:]
        labels_train = [0]*len(training_images)
        labels_test = [0]
        training_tensorList_and_labels, test_tensorList_and_labels = self.dataLoader.augment_and_create_dataset(training_images, labels_train, test_images, labels_test)
        train_batch, test_batch = self.dataLoader.create_mini_batches(training_tensorList_and_labels,test_tensorList_and_labels)        
        new_model = self.takeGradientSteps(train_batch,mode='testing')
        loss, accuracy = self.evaluateModel(new_model,test_batch)
        print(loss, accuracy)
        return loss, accuracy
    
    #This is the main method that runs the experiment. 
    def run(self):
        self.model.to(self.device)
        #check if path exists and checkpoint exists and reload everything
        self.model.train()    
        while self.curr_iter < self.meta_iter:
            #take k gradient steps
            train_batch, val_batch = self.dataLoader.sample_task_from_training_and_val()
            newModel = self.takeGradientSteps(train_batch)
            #from new model update model with new params 
            for currParams, newParams in zip(self.model.parameters(), newModel.parameters()):
                if currParams.grad is None: #if gradient doesn't exist or not initialized 
                    currParams.grad = Variable(torch.zeros(newParams.size())).to(self.device) #initialize the params for update
                currParams.grad.data.add_(self.epsilon*(currParams.data - newParams.data)) #(update)
            #update the model parameters, take step first 
            #as loss is based on the loss from learner
            self.meta_optimizer.step()
            #set optimizer to zero for next iteration of learning 
            self.meta_optimizer.zero_grad()

            #during validation, we are testing to see how well the model adjusts to an unseen task after 
            #k gradient steps
            if self.curr_iter % 1000 == 0:
                #evaluate meta learning loss on training and validation set
                training_stat = self.evaluateModel(self.model, train_batch)
                self.training_loss.append(training_stat[0])
                newModel = self.takeGradientSteps(val_batch)
                validation_stat = self.evaluateModel(newModel, val_batch)
                self.validation_loss.append(validation_stat[0])
                print('iteration ' + str(self.curr_iter) + " avg training_loss: " + str(np.mean(self.training_loss)))
                print('iteration ' + str(self.curr_iter) + " avg validation_loss: " + str(np.mean(self.validation_loss)))
                self.saveState()
            self.curr_iter+=1
        print('done training!')
        return self.model     

    #This method performs the fine tuning. In this method, we get a chosen_task 
    #and perform k steps of SGD. We first instantiate a neural network and transfer 
    #the weights of the meta learner's weights to this model. We then perform 10 steps 
    #of SGD from which the neural network with weights W becomes a neural network with weights 
    #phi. We return the neural network with weights phi. 
    def takeGradientSteps(self, chosen_batch, mode='training'):
        if mode == 'training':
            K = self.totalGradSteps
        elif mode == 'testing':
            K = 50
        new_model = OmniglotModel(self.model.num_classes).to(self.device)
        new_model.train()
        #new model parameters equal to initial model parameters 
        new_model.load_state_dict(self.model.state_dict()) 
        inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=self.lr_inner)
        #sample batch of tasks 
        #take K gradient steps and perform SGD to update parameters 
        for k in range(K): 
            for batch, labels in chosen_batch: 
                batch_var = Variable(batch).to(self.device)
                label_var = Variable(labels).to(self.device)
                label_pred = new_model(batch_var)
                cross_entropy_loss = self.loss_func(label_pred,label_var)
                inner_optimizer.zero_grad()
                cross_entropy_loss.backward()
                inner_optimizer.step()
        return new_model




DATA_PATH = os.getcwd()
print(DATA_PATH)
model = OmniglotModel(5)
lr_inner = 1e-3
lr_outer = 1 
dataLoader = OmniglotDataLoader(root_dir=DATA_PATH, K=5, N=5)
meta_iter = 100000
K = 5
checkpoint_dir = 'exp1'
exp = ReptileExperiment(model, lr_inner, lr_outer, dataLoader, meta_iter, K, checkpoint_dir)
exp.run()