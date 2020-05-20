import torch
import torch.backends.cudnn as cudnn
import cv2
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import datetime  

import torch.nn as nn

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl
import datetime 
 
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
torch.cuda.set_device(0)
root="c:/breastcancer/breakhis/BreaKHis_v1/histology_slides/"
savepath="c:/breastcancer/breakhis/BreaKHis_v1/histology_slides/inde/"

def saveprobdata1(savefilename,  probdata):
    
    target = open(savefilename, 'a')
    len1=np.shape(probdata)[0]
    for i in range(len1):
        label=int(probdata[i,0])
        prob="%.7f" % probdata[i,1] 
        
        s=str(label) + "\t"+str(prob)+"\n"

        target.write(s)
    target.close()
def ROC( y_test,y_predicted):
    y = y_test
    pred =y_predicted
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    v_auc=auc(fpr, tpr)
    print(v_auc)
    #x = [_v[0] for _v in xy_arr]
    #y = [_v[1] for _v in xy_arr]
    pl.title("ROC curve of %s (AUC = %.4f)" % ('M6A' , v_auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot( fpr ,tpr)
    pl.show()

    return v_auc

def savelossaccdata(savefilename,lossdata):
    
    target = open(savefilename, 'a')
    
    len1=len(lossdata[:,0])
    s ="epoch\ttrainloss\trainacc\ttestloss\ttestacc\n"
    target.write(s)        
    
    for i in range(len1):         
        s=str("%.0f" %lossdata[i,0])+"\t"+str("%.7f" %lossdata[i,1]) +"\t"+str("%.7f" %lossdata[i,2]) +"\t"+ str("%.7f" %lossdata[i,3]) +"\t"+str("%.7f" %lossdata[i,4]) 
        s+="\n"
        target.write(s)
    target.close()     
def saveauc(savefilename,cvnum,i,kk,  auc):
    target = open(savefilename, 'a')
    #s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    #target.write(s0)    
    s1=str(cvnum)+"\t"+str(i)+"\t"+str(kk)+"\t"+ str(auc)+"\n"
    target.write(s1)
    target.close()   

def savelabelenumdata(savefilename,  probdata):
    
    target = open(savefilename, 'a')
    len1=np.shape(probdata)[0]
    for i in range(len1):
        label=int(probdata[i,0])
        pos="%.0f" % probdata[i,1] 
        neg="%.0f" % probdata[i,2] 
        acc="%.4f" % probdata[i,3] 
        s=str(label) + "\t"+str(pos)+ "\t"+str(neg)+ "\t"+str(acc) +"\n"

        target.write(s)
    target.close()
    
def saveresolabelenumdata(savefilename, resodict):
    
    target = open(savefilename, 'a')
   
    for key in resodict:
        probdata=resodict[key]
        len1=np.shape(probdata)[0]
        s1=np.sum(probdata[:,1])
        s2=np.sum(probdata[:,2])
        s3=(s1/(s1+s2))
        s=key+"\t" +"\t" +str(s1)+"\t"+str(s2)+"\t"+str("%.4f"  % s3)+"\n"
        target.write(s)
        for i in range(len1):
            label=int(probdata[i,0])
            pos="%.0f" % probdata[i,1] 
            neg="%.0f" % probdata[i,2] 
            acc="%.4f" % probdata[i,3] 
            s="\t"+str(label) + "\t"+str(pos)+ "\t"+str(neg)+ "\t"+str(acc) +"\n"
    
            target.write(s)
    target.close()    
def savecrosspreddata(savefilename, crossdata):
    
    target = open(savefilename, 'a')
    len1=2
    len2=2
    for i in range(len1): 
        s=""
        
        for j in range(len2): 
            s+= str("%.0f" %crossdata[i,j])+"\t"
        s+="\n"
        target.write(s)
    target.close()   
    
def saveprobdata(savefilename,probdata,test_infor):
    
    target = open(savefilename, 'a')
    
    len1=len(probdata[:,0])
        
    
    for i in range(len1):         
        s=str("%.0f" %probdata[i,0])+"\t"+str("%.0f" %probdata[i,1])+"\t"+  test_infor[i]  
        s+="\n"
        target.write(s)
    target.close()        
    
    
def saveauchead(savefilename ):
    target = open(savefilename, 'a')
    s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    target.write(s0)    
    
    target.close() 
    
def saveteststr(savefilename,s ):
    target = open(savefilename, 'a')
    
    target.write(s)    
    
    target.close()  

# -----------------ready the dataset--------------------------
def opencvLoad(imgPath,resizeH,resizeW):
    image = cv2.imread(imgPath)
    #print(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))  
    image = torch.from_numpy(image)
    return image
    
class LoadPartDataset(Dataset):
    def __init__(self, txt):
        imgs = []
        pt=txt+".txt"
        fh = open(pt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            labelList = int(words[0])
            imageList =root+ words[1]
            imgs.append((imageList, labelList))
 
        self.imgs = imgs
            
    def __getitem__(self, item):
        image, label = self.imgs[item]
        #print(image)
        img = opencvLoad(image,227,227)
        return img,label
    def __len__(self):
        return len(self.imgs)
        
def loadTrainData(txt=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        label = int(words[0])
        image = cv2.imread(root+words[1])
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 1, 0))  
        image = torch.from_numpy(image)
        imgs.append((image, label))
    return imgs

def loadTestData(txt=None):
    
    imgs = []
    
    pt=txt+".txt"
    fh = open(pt, 'r')
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        
        imageList = words[1]
        imgs.append( imageList )
         
    fh.close()
 
    
    return imgs 
            
# trainSet=loadTrainData(txt=root+'train.txt')
# test_data=loadTrainData(txt=root+'train.txt')
trainSet =LoadPartDataset(txt=root+'train_bin')
test_data=LoadPartDataset(txt=root+'test_bin')
len_test=len(test_data.imgs)
test_infor =loadTestData(txt=root+'test_bin')
train_loader = DataLoader(dataset=trainSet, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=10)
 

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [ torch. nn.ReflectionPad2d(1),
                       torch. nn.Conv2d(in_features, in_features, 3),
                       torch. nn.InstanceNorm2d(in_features),
                       torch. nn.ReLU(inplace=True),
                       torch. nn.ReflectionPad2d(1),
                       torch. nn.Conv2d(in_features, in_features,3),
                       torch. nn.InstanceNorm2d(in_features)  ]

        self.conv_block =torch. nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)  


#定义conv-bn-relu函数
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    conv = torch.nn.Sequential(
      torch.  nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
       torch. nn.BatchNorm2d(out_channel, eps=1e-3),
       torch. nn.ReLU(True),
    )
    return conv

#定义incepion结构，见inception图
class inception(torch.nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5,
                 out4_1):
        super(inception, self).__init__()
        self.branch1 = conv_relu(in_channel, out1_1, 1)
        self.branch2 = torch.nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch3 = torch.nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output 
 
#-----------------create the Net and training------------------------
 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
      
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(21, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
         
        self.res1=ResidualBlock(21 )
 
 
 
        
        self.inc1=inception( 3,4, 4, 12, 1, 3,2)     
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 512),#9216 21*6*6
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )
    
 
        
 
    def forward(self, x):
  
        
       
        
     
        dd=self.inc1(x) 
      
 
        dd=self.res1(dd)
     
        conv1_out = self.conv1(dd)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)  
 
        res = conv5_out.view( conv5_out.size(0), -1) 
 
        out = self.dense(res)
 
        return out
 
 
model = Net()
 
finetune = None
#finetune = root+'model/_iter_99.pth'
 
if finetune is not None:
    print( '[0] Load Model {}'.format(finetune))
 
    pretrained_dict = model.state_dict()
    finetune_dict = torch.load(finetune)
 
    # model_dict = torch.load(finetune)
    # pretrained_dict = net.state_dict()
 
    model_dict = {k: v for k, v in finetune_dict.items() if k in pretrained_dict}
    pretrained_dict.update(model_dict)
 
    model.load_state_dict(pretrained_dict)
 
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()
cudnn.benchmark = True
print(model)
 
 
#optimizer = torch.optim.Adam(model.parameters())
#loss_func = torch.nn.CrossEntropyLoss()

# updata net
epochnum=100
loss_acc_mat= np.zeros((epochnum,5),dtype=np.float32)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (model.parameters(), lr=0.0001  )
#optimizer = torch.optim.SGD(list(model.parameters())[:], lr=lr , momentum=0.9)
model.train() 
auclist=dict()
for epoch in range(epochnum):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for trainData, trainLabel in train_loader:
        trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
        optimizer.zero_grad()
        out = model(trainData)
        loss = loss_func(out, trainLabel)
        #print(loss)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == trainLabel).sum()
        train_acc += train_correct.item()
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #  if epoch % 100 == 0:
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    print(now_time,'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        trainSet)), train_acc / (len(trainSet))))
 
    loss_acc_mat[epoch,1]=    train_loss/(len(
        trainSet))
    loss_acc_mat[epoch,2]=    train_acc/(len(
        trainSet)) 
    #if (epoch + 1) % 10 == 0:
    #    sodir = './model/_iter_{}.pth'.format(epoch)
    #    print ('[5] Model save {}'.format(sodir))
    #torch.save(model.module.state_dict(), sodir)
 
    # adjust
    #if (epoch + 1)% 100 == 0:
    #      lr = lr / 10
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
 
    #evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    y_test_com = np.zeros((len_test,1),dtype=np.float32)
    y_predicted_com = np.zeros((len_test,1),dtype=np.float32)
    y_test_com_l = None
    y_predicted_com_l = None
    pos_test=0
    kk=0
    for testData,testLabel in test_loader:
         testData, testLabel = Variable(testData.cuda()), Variable(testLabel.cuda())
         out = model( testData)
         loss = loss_func(out, testLabel )
 
         len1=len( testLabel )
         y_test=testLabel .reshape((len1,1))


         
         #print(loss)
         eval_loss += loss.item()
         pred = torch.max(out, 1)[1]
         num_correct = (pred == testLabel).sum()
         eval_acc += num_correct.item()

         y_test=testLabel.cpu() .detach().numpy()     
         y_test=y_test.reshape((len1,1))  
         y_predicted_l= pred.cpu() .detach().numpy()        
         y_predicted_l=y_predicted_l.reshape((len1,1))  
         if kk==0:
             y_predicted_com_l =y_predicted_l
             y_test_com_l=y_test
         else:
             y_predicted_com_l =np.vstack((y_predicted_com_l ,y_predicted_l))
             y_test_com_l=  np.vstack((y_test_com_l ,y_test))             
             
         kk +=1 
         

   
   
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    acc=int((eval_acc /  len(test_data))*10000)   
    
    print(now_time,'Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
         test_data)), eval_acc / (len(test_data))))
 
    loss_acc_mat[epoch,3]=    eval_loss / (len(test_data))
    loss_acc_mat[epoch,4]=    eval_acc  / (len(test_data)) 
 
    ##cross 
    np_data_full_l=np.hstack((y_test_com_l ,y_predicted_com_l ))    
    len1=len(y_test_com_l)
    pred_ct = np.zeros((8,4),dtype=np.float32)
    tt=0
    for i in range(8):
        pred_ct[i,0] =tt
     
        tt+=1
    ##
    for i in range(len1):
        label=int(y_test_com_l[i])
        pred=int(y_predicted_com_l[i]) 
       
        if pred==label:
            pred_ct[label,1] =pred_ct[label,1]+1 
        else:
            pred_ct[label,2] =pred_ct[label,2]+1 
 
    pred_ct[:,3]   =pred_ct[:,1]/(pred_ct[:,1]+pred_ct[:,2])    
    savelabelenumdata(savepath+str( acc)+"_pred_"+str(epoch)+".txt", pred_ct)
    saveprobdata(savepath+str( acc)+"_labelclass_"+str(epoch)+".txt", np_data_full_l,test_infor)  
    
  
    model.train()
    model.zero_grad()                   
savelossaccdata(savepath +"lossacc.txt", loss_acc_mat)      