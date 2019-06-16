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
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpointPath = os.path.join(checkpoint_dir,"checkpoint.pth.tar")
        self.config_path =  os.path.join(checkpoint_dir,"config.txt")
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
                    
    def saveState(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpointPath)
        with open(self.config_path, 'w+') as f:
            print(self, file=f)
    
    def loadState(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    
    def checkpoint_exists(self):
        return os.path.exists(self.checkpointPath) and os.path.exists(self.config_path)
        
    def evaluateModel(self, model, chosen_batch, mode='during_training'):
        self.model.eval()
        loss = 0 
        with torch.no_grad(): 
            for batch, labels in chosen_batch:
                batch_var = Variable(batch.to(self.device))
                label_var = Variable(labels.to(self.device))
                label_pred = model(batch_var)
                cross_entropy_loss = self.loss_func(label_pred,label_var)
                loss += cross_entropy_loss.item()        
        if mode == 'during_training':
            self.model.train()
        return loss

    def run(self):
        #check if path exists and checkpoint exists and reload everything
        self.model.train()    
        while self.curr_iter < self.meta_iter:
            #take k gradient steps
            train_batch, val_batch = self.dataLoader.sample_task_from_training_and_val()
            newModel = self.takeGradientSteps(train_batch)
            #from new model update model with new params
            for currParams, newParams in zip(self.model.parameters(), newModel.parameters()):
                if currParams.grad is None: #if gradient doesn't exist or not initialized 
                    currParams.grad = Variable(torch.zeros(newParams.size())) #initialize the params for update
                currParams.grad.data.add_(self.epsilon*(currParams.data - newParams.data)) #(update)
            #update the model parameters, take step first 
            #as loss is based on the loss from learner
            self.meta_optimizer.step()
            #set optimizer to zero for next iteration of learning 
            self.meta_optimizer.zero_grad()
            #evaluate meta learning loss on training and validation set
            self.training_loss.append(self.evaluateModel(newModel, train_batch))
            
            newModel = self.takeGradientSteps(val_batch)
            self.validation_loss.append(self.evaluateModel(newModel, val_batch))
            #during validation, we are testing to see how well the model adjusts to an unseen task after 
            #k gradient steps
            if self.curr_iter % 1 == 0:
                print('iteration ' + str(self.curr_iter) + " avg training_loss: " + str(np.mean(self.training_loss)))
                print('iteration ' + str(self.curr_iter) + " avg validation_loss: " + str(np.mean(self.validation_loss)))
                self.saveState()
            self.curr_iter+=1
        print('done training!')
        return self.model     

    def takeGradientSteps(self, chosen_batch):
        new_model = OmniglotModel(self.model.num_classes)
        #new model parameters equal to initial model parameters 
        new_model.load_state_dict(self.model.state_dict()) 
        inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=self.lr_inner)
        #sample batch of tasks 
        #take K gradient steps and perform SGD to update parameters 
        for k in range(self.totalGradSteps): 
            for batch, labels in chosen_batch: 
                batch_var = Variable(batch.to(self.device))
                label_var = Variable(labels.to(self.device))
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