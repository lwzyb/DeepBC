# -*- coding: utf-8 -*-
"""

"""

#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
 
 
from matplotlib.ticker import MultipleLocator
 

def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    #linesum = matrix.sum(1)
    #linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    #matrix /= linesum
    # plot
    plt.switch_backend('agg')

    font = {'family' : 'Times New Roman',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 40,  
        }
    font2 = {'family' : 'Times New Roman',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 50,  
        }  
    fig = plt.figure(figsize=[10,10] )
    ax = fig.add_subplot(111)
   
    cax = ax.matshow(matrix,cmap=plt.cm.Oranges )#hot_r)
    cb=fig.colorbar(cax )
    cb.ax.tick_params(labelsize=30)  #设置色标刻度字体大小。
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i, j, str('%3.0f' % (matrix[  j,i] )), va='center', ha='center',fontdict=font)#matrix[i, i]* 100
    ax.set_xticklabels([''] + classes, rotation=90,fontdict=font)#, rotation=90
    ax.set_yticklabels([''] + classes,fontdict=font)
    ax.set_xlabel("Prediction",fontdict=font2)
    ax.set_ylabel("Real Label",fontdict=font2)
    #cb.set_label('colorbar',fontdict=font2) #设置colorbar的标签字体及其大小 
    #ax.tick_params(labelsize=40)
    #ax.tick_params(axis='both', which='major', labelsize=20)
    """
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        #tick.label.set_rotation('vertical')     
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        #tick.label.set_rotation('vertical')
    """
    #save
    plt.savefig(savname,dpi=400,bbox_inches = 'tight')
 

savname="C:/breastcancer/breakhis/BreaKHis_v1/histology_slides/ConfuMatrix_DeepBC.png"
confusionfile="C:/breastcancer/breakhis/BreaKHis_v1/histology_slides/ConfuMatrix_DeepBC.txt"
labelfile="C:/breastcancer/breakhis/BreaKHis_v1/histology_slides/labelnum.txt"

 
 
 
def readconfusion():
    fh = open(confusionfile, 'r')
    cm=np.zeros((labelnum,labelnum),dtype=np.float32)
    i=0
    for line in fh:
        line = line.strip('\n')
        ss=line.split("\t")
         
        len1=len(ss)
       
        for j in range(len1):
            if ss[j]=="":
                continue
            
            cm[i,j]=float(int(ss[j] ))
        
        i+=1
    fh.close()    
    return cm

def readlabelfile(labelfile):
    
    fo=open( labelfile ,"r")
    ls=fo.readlines() 
    kk=0
    labelfolderlist=list()
    for s in ls:
 
        #print(s)
        split=s.split("\t")
        len2=len(split)
        if len2>0:
            
            folder1=  split[1].replace("\n","") 
            labelfolderlist.append(folder1)
             
        kk +=1
      
    fo.close()
    return labelfolderlist


classes= readlabelfile(labelfile)
labelnum=len(classes)
print(classes)
confusion_matrix =readconfusion()
plotCM(classes, confusion_matrix, savname)