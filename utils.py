import numpy as np
import time
import pickle
import os
from datetime import datetime
import torch
import torch.cuda
import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from util import performance_eval 


#################################
#### machine learning model #####
#################################
class MLModel ():
    '''
    This class is implemented for the automated lerning of ML models provided from Sci-kit Learn.
    '''
    def __init__ (self,  model, num_classes, note= ''):
        self.m = model
        self.y_pred_prob = None
        self.coef = None     
        self.auroc = {}
        self.auprc = {}
        self.note =note+ datetime.today().strftime("%Y%m%d_%H%M_%S")
        self.name = model.__class__.__name__+'_'+ note
        self.num_classes = num_classes

        print('='*50) 
        print('Model Info:', self.m)   
    def execution (self, X_train, y_train, X_test, y_test): 
        print('~'*20,'Training','~'*20)
        t = time.process_time()
        self.m = self.m.fit(X_train, y_train)
        print('*** Complete! Training time:', '{:.3f}'.format(time.process_time() - t),'sec ***')

        print('~'*20,'Testing','~'*20)   
        t = time.process_time()
        self.y_pred_prob = self.m.predict_proba(X_test)
        if self.name.find('XG')>-1:
            self.coef = self.m.feature_importances_
        else:
            self.coef = self.m.coef_[0]
        performance_eval.result_report(self,y_test)
        elapsed_time = time.process_time() - t
        print('***','Complete! Testing time:', '{:.3f}'.format(time.process_time() - t),'sec ***\n')
            

##############################
#### deep learning model #####
##############################
class DeepModel():
    '''
    This class is implemented for the automated lerning of own deep learning models.
    '''
    def __init__(self, model, num_classes=1, num_outputs =1, 
                criterion= torch.nn.CrossEntropyLoss(), lr=1e-3, batch_size = 32, epoch=10,
                num_workers =8, verbose= True, note =''):

        self.m = model_on_GPU(model) 

        ## learning parameters                
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch       
        self.criterion, self.cri_name = criterion_on_GPU(criterion)
        self.optimizer = torch.optim.Adam(self.m.parameters(), lr=lr, weight_decay=1e-5,)
        
        ## excution parameters
        self.num_classes=num_classes
        self.num_outputs = num_outputs
        self.num_workers = num_workers
        self.verbose = verbose
        
        ## log parameters
        self.note =note+ datetime.today().strftime("%Y%m%d_%H%M_%S")
        self.name = model._get_name()+'_'+self.note 
        self.att_score = None
        self.y_pred_prob = None
        self.loss_record = []
        self.auroc = dict()
        self.auprc = dict()

            
    def execution (self, train_set, test_set): 
        print('~'*20,'Training','~'*20)
        t= time.process_time()  
        self.training(train_set)   
        print('*** Complete! Training time:', '{:.3f}'.format(time.process_time() - t),'sec ***')

        print('~'*20,'Testing','~'*20)   
        t= time.process_time()  
        self.test(test_set)
        print('***','Complete! Testing time:', '{:.3f}'.format(time.process_time() - t),'sec ***\n')
      

    def test(self, test_set):
        self.y_pred_prob = None ## 예측 기록지
        y_test = None ## 정답지
                
        if self.batch_size>test_set.len:
            self.batch_size=test_set.len 
            
        self.m.eval()
        deep_test_set = DataLoader(dataset=test_set, batch_size = self.batch_size, num_workers = self.num_workers, drop_last=True)     
        with torch.no_grad():
            for curr_batch, (x, y) in enumerate(deep_test_set, 1):
                x = to_GPU(x)
                y_hat = self.m.forward(x)
                if self.num_outputs>1:
                    y_hat = y_hat[0]   
                if self.y_pred_prob is None:
                    self.y_pred_prob = y_hat.cpu().detach().numpy()
                    y_test = y.cpu().detach().numpy()
                else:
                    self.y_pred_prob = np.append(self.y_pred_prob, y_hat.cpu().detach().numpy(), axis=0)
                    y_test = np.append(y_test, y.cpu().detach().numpy(), axis=0)

        if y_test is not None:
            performance_eval.result_report(self,y_test.astype(int)) 

    def training(self, train_set,save=True):   
        if self.batch_size>train_set.len:
            self.batch_size=int(train_set.len/4)

        total_batchs = int(train_set.len/self.batch_size)    
        es_iter = EarlyStopping(patience=5) ## 전체 Iter가 자주 튀는 경우를 잡아보자 
        deep_train_set = DataLoader(dataset=train_set,batch_size = self.batch_size, shuffle=True, num_workers =self.num_workers, drop_last=True) 
      
        for it in range(self.epoch):
            epoch_loss=0
            y_hat_list =[] 
            y_list =[]
            prog_checker = ProgressReport(total_batchs)   
            for curr_batch, (x, y) in enumerate(deep_train_set, 1):         
                x = to_GPU(x)
                y_hat = self.m.forward(x)
                if self.num_outputs>1:
                    self.att_score = (to_CPU(y_hat[1]).numpy()[:-train_set.num_features]) ## for the attention score
                    y_hat = y_hat[0] 
                y = y_type_changer(y,self.cri_name)    
                batch_loss = self.criterion(y_hat, y)
                
                ## backpropagation
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            
                ## log per batch
                y_hat_list.extend(to_CPU(y_hat).numpy())
                y_list.extend(to_CPU(y).numpy())        
                epoch_loss += batch_loss.item()
                if self.verbose:
                    prog_checker(it, curr_batch, batch_loss, epoch_loss, np.array(y_hat_list), np.array(y_list),self.cri_name,self.m,self.name)   
                
            self.loss_record.append(epoch_loss)
            if es_iter.validate(epoch_loss): break  
        self.epoch = it # final epoch records  

########################################################################
### For early stopping the leaning ######
class EarlyStopping():
    def __init__(self, patience=5, threshold = 0.01):
        self._step = 0
        #self._zero = 0
        self._same = int(patience/2) ## 거의 변화가 없는 경우
        self.threshold = threshold
        self._loss = float('inf')
        self.patience  = patience
 
    def validate(self, loss):
        #print(self._loss, loss)
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                print('Early stopped by the instability....')
                return True
        if abs(loss - self._loss) < self.threshold:
            self._same += 1
            if self._same > self.patience:
                print('Early stopped by unchangeability...')
                return True
        else:
            self._loss = loss # loss update 
            #self._step = 0
        return False

class ProgressReport(object):
    def __init__(self, total_batchs, chk_rate = 10):
        self.final_batch = total_batchs
        self.chk_point = []
        self.rate = 1
        if total_batchs > chk_rate: 
            hop = total_batchs/chk_rate
            for i in range(1,int(100/chk_rate)+1):
                self.chk_point.append(int(i*hop))
        else:
            for i in range(1,total_batchs+1):
                self.chk_point.append(i) 
    def chk_overfitting(self, pred, true):
        n_classes = 2
        if pred.shape[1] == 1:
            pred = np.squeeze(pred) 
        else:
            n_classes = int(pred.shape[1]) # num_variables
            true = np.eye(n_classes)[true] 

        pred[pred>=((n_classes-1)/n_classes)]=1
        pred[pred!=1]=0
        return precision_score(true, pred,average='weighted'), recall_score(true, pred,average='weighted')

    def __call__(self, it, curr_batch, loss, epoch_loss,y_hat,y,cri_type, m,name):
        precsion = np.nan
        recall = np.nan
        if str(cri_type) is not 'MSELoss()':
            precsion, recall = self.chk_overfitting(y_hat,y)

        if curr_batch == self.chk_point[-1]:
            print('{:,}th {:}%| total_loss: {:.4f} | Precision:{:.4f} | Recall:{:.4f}'.format(it,100, epoch_loss, precsion, recall))
        
        elif curr_batch == self.chk_point[self.rate-1]:
            print('{:,}th {:}%| loss: {:.4f} | Precision:{:.4f} | Recall:{:.4f}'.format(it,self.rate*10, loss, precsion, recall))
            self.rate+=1
            
        ## save 
        if (curr_batch == self.chk_point[int(len(self.chk_point)/2)])|(curr_batch == self.chk_point[-1]):
            if not os.path.exists('./model/'): os.makedirs('./model/')
            if torch.cuda.is_available() == True:
                torch.save(m.module,'./model/'+name)
                with open('./model/'+name+'_param', 'wb') as f:
                    pickle.dump(m.module, f)
            else:
                torch.save(m,'./model/'+name)
                with open('./model/'+name+'_param', 'wb') as f:
                    pickle.dump(m, f)


def y_type_changer(y,cri_type):
    if str(cri_type) == 'CrossEntropyLoss':
        return to_GPU(y.long())
    elif str(cri_type) == 'BCELoss':
        return to_GPU(y.float().reshape(-1,1))  
    elif str(cri_type) == 'MSELoss':
        return to_GPU(y.float()) 

def initialize_weights(net):
    torch.manual_seed(1)
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass

            
########################################################################
################################## GPU mode ############################
########################################################################
def to_GPU(tensor):
    if torch.cuda.is_available():
       # print("Variables on Device memory")
        return tensor.cuda()
    else:
       # print("Variables on Host memory")
        return tensor
    
def to_CPU(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach()
    else:
        return tensor
    
def model_on_GPU (model,multi=True):
    
    print('='*50) 
    print('Model Info:', model)   
    
    if ((multi==True)& (torch.cuda.device_count() > 1)):
        print(torch.cuda.device_count(), 'GPUs used!')
        #model = DataParallelModel(model)
        model = torch.nn.DataParallel(model)
        model.cuda()

    elif torch.cuda.is_available():
        print("Model on Device memory")
        model.cuda()
    else:
        print("Model on Host memory")
       
    return model      


def criterion_on_GPU (criterion):
    cri_name = criterion._get_name()
    #print(cri_name)
    if torch.cuda.device_count() > 10: 
        print('Loss fucntion parallelized')
        criterion = DataParallelCriterion(criterion)
        criterion.cuda()
    return criterion, cri_name