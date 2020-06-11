# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:20:17 2020

@author: ASUS
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.fixes import signature
from sklearn.metrics import auc
import math

plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
def readprob(probfile):
    
     
    img_true_label=dict()
    img_pred_label=dict()    
    fh = open(probfile, 'r')
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        ss = line.split("\t")
        key=ss[2]
        img_true_label[key]=int(ss[0])
        img_pred_label[key]=int(ss[1])
 
         
    fh.close()
 
    
    return img_true_label,img_pred_label


def calfullacc(img_true_label,img_pred_label):
    
    num=len(img_true_label)
    prob=np.zeros((num,2),dtype=np.float32)
    pat_count_dict=dict()
    pat_correct_dict=dict()    
    i=0
    for key in img_true_label:
        split1=key.split("/")
        patientid=split1[4]
        pat_count=pat_count_dict.get( patientid,0)
        pat_count+=1
        pat_count_dict[patientid]=pat_count

        true_label=img_true_label[key]
        pred_label=img_pred_label[key]
        prob[i,0]=true_label
        prob[i,1]=pred_label
        
        pat_correct=pat_correct_dict.get( patientid,0)
        if true_label== pred_label:
            pat_correct+=1    
        pat_correct_dict[patientid]=pat_correct        
        i=i+1
 
    pat_num=   len(pat_count_dict)    
    acc_pat=0
    for key in pat_count_dict:
         pat_count=pat_count_dict.get( patientid,0)
         pat_correct=pat_correct_dict.get( patientid,0)
         acc_pat_key=pat_correct/pat_count
         acc_pat+=acc_pat_key
         
    acc_pat_f=acc_pat/pat_num
    acc_img_f =accuracy_score( prob[:,0],  prob[:,1] )
      
    #f1_score
    f1score_f=f1_score( prob[:,0],  prob[:,1])    


    tn, fp, fn, tp = confusion_matrix(prob[:,0],  prob[:,1]).ravel()    
    
    return acc_img_f,acc_pat_f,f1score_f,tn, fp, fn, tp


def calzoomacc(img_true_label,img_pred_label,zoom):
    
 
    
    pat_count_dict=dict()
    pat_correct_dict=dict()   
    pred_list =list()
    label_list =list()
    i=0
    for key in img_true_label:
        split1=key.split("/")
        patientid=split1[4]
        zoomid=split1[5]      
        if zoomid !=zoom:
            continue
        
        pat_count=pat_count_dict.get( patientid,0)
        pat_count+=1
        pat_count_dict[patientid]=pat_count

        true_label=img_true_label[key]
        pred_label=img_pred_label[key]
 
        pred_list.append(pred_label)
        label_list.append(true_label)        
        pat_correct=pat_correct_dict.get( patientid,0)
        if true_label== pred_label:
            pat_correct+=1    
        pat_correct_dict[patientid]=pat_correct        
        i=i+1
    prob1=np.array(pred_list) 
    prob0=np.array(label_list) 
    pat_num=   len(pat_count_dict)    
    acc_pat=0
    for key in pat_count_dict:
         pat_count=pat_count_dict.get( patientid,0)
         pat_correct=pat_correct_dict.get( patientid,0)
         acc_pat_key=pat_correct/pat_count
         acc_pat+=acc_pat_key
         
    acc_pat_f=acc_pat/pat_num
    acc_img_f =accuracy_score( prob0,  prob1 )
      
    #f1_score
    f1score_f=f1_score( prob0,  prob1)    

 
    
    return acc_img_f,acc_pat_f,f1score_f 


def calbinacc(img_true_label,img_pred_label ):
    
   
    ftype_erro_dict=dict()
    ftype_correct_dict=dict()   
   
    i=0
    for key in img_true_label:
        split1=key.split("/") 
        ftypeid=split1[1]
        subftypeid=split1[3]     
        ftypeid=ftypeid+"_"+subftypeid
        true_label=img_true_label[key]
        pred_label=img_pred_label[key] 
        
        pat_correct=ftype_correct_dict.get( ftypeid,0)
        pat_erro=ftype_erro_dict.get( ftypeid,0)
        
        if true_label== pred_label:
            pat_correct+=1
            ftype_correct_dict[ftypeid]=pat_correct
        else:
            pat_erro+=1
            ftype_erro_dict[ftypeid]=pat_erro        
        i=i+1
    
    return ftype_erro_dict, ftype_correct_dict

probfile="C:/breastcancer/breakhis/BreaKHis_v1/histology_slides/inde -- VGG16/8658_prob_29.txt"
metricfullfile="C:/breastcancer/breakhis/BreaKHis_v1/histology_slides/metric_VGG16.txt"
if __name__ == "__main__":    
     img_true_label,img_pred_label=readprob(probfile)
     acc_img_f,acc_pat_f,f1score_f,tn_f, fp_f, fn_f, tp_f=calfullacc(img_true_label,img_pred_label)
     ftype_erro_dict, ftype_correct_dict=calbinacc(img_true_label,img_pred_label )
     #print( tn_f, fp_f, fn_f, tp_f)
     #print(acc_img_f,acc_pat_f,f1score_f )
     acc_img_40X,acc_pat_40X,f1score_40X=calzoomacc(img_true_label,img_pred_label,'40X')
     #print( acc_40X,acc_pat_40X,f1score_40X )     
     acc_img_100X,acc_pat_100X,f1score_100X=calzoomacc(img_true_label,img_pred_label,'100X')
     #print( acc_100X,acc_pat_100X,f1score_100X )
          
     acc_img_200X,acc_pat_200X,f1score_200X=calzoomacc(img_true_label,img_pred_label,'200X')
     #print( acc_200X,acc_pat_200X,f1score_200X )
     
     acc_img_400X,acc_pat_400X,f1score_400X=calzoomacc(img_true_label,img_pred_label,'400X')
     #print( acc_400X,acc_pat_400X,f1score_400X )     
     
     target = open(metricfullfile, 'a')
   
     s="tn\tfp\tfn\ttp\n"
     target.write(s)   
    
      
     s=str("%.0f" % tn_f)+"\t"+ str("%.0f" % fp_f)+"\t"+str("%.0f" % fn_f)+"\t"+str("%.0f" %  tp_f) +"\n"
     target.write(s)   
     s="zoom\tacc_img\tacc_pat\tf1score\n"
     target.write(s)    
     s="Full\t"+str("%.4f" % acc_img_f)+"\t"+ str("%.4f" % acc_pat_f)+"\t"+str("%.4f" % f1score_f) +"\n"
   
     target.write(s)    
     s="40X\t"+str("%.4f" % acc_img_40X)+"\t"+ str("%.4f" % acc_pat_40X)+"\t"+str("%.4f" % f1score_40X) +"\n"
 
     target.write(s)    
     s="100X\t"+str("%.4f" % acc_img_100X)+"\t"+ str("%.4f" % acc_pat_100X)+"\t"+str("%.4f" % f1score_100X) +"\n"
 
     target.write(s)    
     s="200X\t"+str("%.4f" % acc_img_200X)+"\t"+ str("%.4f" % acc_pat_200X)+"\t"+str("%.4f" % f1score_200X) +"\n"
 
     target.write(s)    
     s="400X\t"+str("%.4f" % acc_img_400X)+"\t"+ str("%.4f" % acc_pat_400X)+"\t"+str("%.4f" % f1score_400X) +"\n"
     target.write(s)   
      
     s1="content"
     s2="correct"
     s3="error"
     s4="errorrate"
     for key in ftype_correct_dict:
         correctnum=ftype_correct_dict[key]
         errornum= ftype_erro_dict[key]
         errorrate=errornum/(errornum+correctnum)
         s1+= "\t"+key 
         s2+= "\t"+str("%.0f" % correctnum  )
         s3+=  "\t"+str("%.0f" % errornum )   
         s4+=  "\t"+str("%.4f" % errorrate )   
    
     ss=s1+"\n"+s2+"\n"+s3+"\n"+s4+"\n"
     target.write(ss)   
     target.close()
