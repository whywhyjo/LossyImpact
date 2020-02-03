import pydicom
import os
import sys
sys.path.append('../../../lib')
from net.ResNet import *
from net.CNN import *
from util import batch_data, utils
import torch 
import argparse


def output_size (width,height,k_def,p_def,s_def,num_layer):
    for i in range(num_layer):
        width = int((((width-k_def+(2*p_def))/s_def+1)-k_def)/s_def+1)
        height = int((((height-k_def+(2*p_def))/s_def+1)-k_def)/s_def+1)
    return width, height

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='input: data_path, data_type, data_size,  num_epoch,  ...')
    parser.add_argument('--root_path', type = str, help='including root_path', default='/share_folder/data/breast_sm')
    parser.add_argument('--data_type', type = str, help='lossy or lossless')
    parser.add_argument('--data_size', type = str, help='(0_1/0_3/0_5/0_7/1_0)')
#   parser.add_argument('--num_channel', type = int, help='integer number', default=8)
    parser.add_argument('--num_epoch', type = int, help='integer number', default=10)
#    parser.add_argument('--final_size', type = int, help='integer number')
    parser.add_argument('--train_set', type = str, help='train_list.txt')
    parser.add_argument('--test_set', type = str, help='test_list.txt')
    parser.add_argument('--batch_size', type = int, help='batch size',default=16)
    parser.add_argument('--lr', type = float, help='learning rate',default=1e-3)

    args = parser.parse_args()

    path_dict = {}
    #for dir_name,_,_ in os.walk(os.path.join(args.root_path,args.data_size,args.data_type)): # original
    for dir_name,_,_ in os.walk(os.path.join(args.root_path,args.data_type)):
        try:  
            pt_str = dir_name.split('/')[-1]
        except: pass
        else:
            try: int(pt_str)
            except: 
                pass
            else:
                path_dict[pt_str] = dir_name

    width = 4096 # input width
    height = 3328 # input height
    percent = float(args.data_size.replace('_','.'))
    
    train_cases = []
    with open(args.train_set,'r') as f:
        train_cases = f.read().splitlines()
    #train_set = batch_data.MammoDataset_concat(train_cases,path_dict) ## original
    train_set = batch_data.MammoDataset(train_cases,path_dict,percent)

    test_cases = []
    with open(args.test_set,'r') as f:
        test_cases = f.read().splitlines()       
    #test_set = batch_data.MammoDataset_concat(test_cases,path_dict)
    test_set = batch_data.MammoDataset(test_cases,path_dict,percent)
    

    ch_def = 8 #channel
    k_def = 5 # kernel size
    s_def = 2 # stride size
    p_def = 2 # padding size
       
    if args.data_size == '0_3':
        k_def = 6
        s_def = 2
        p_def = 2
    elif args.data_size == '0_5':
        k_def = 9
        s_def = 2
        p_def = 2
    elif args.data_size == '0_7':
        k_def = 15
        s_def = 2
        p_def = 2
    elif args.data_size == '1_0':
        k_def = 30
        s_def = 2
        p_def = 2
        
    w,h = output_size(int((int(width*percent)+1)*2*0.6),\
                      (int(int(height*percent)+1)*2*0.4),\
                      k_def,s_def,p_def,3)
    final_output = w*h*ch_def*4   
   
    print(final_output)
     
    model = utils.DeepModel(ConvNet_basic_two_fc(ch_size = ch_def, \
                                                k_size = k_def, \
                                                s_size = s_def, \
                                                p_size = p_def,\
                                                final_size = final_output)\
                            , num_classes =1, criterion = torch.nn.BCELoss()\
                            ,lr=args.lr, epoch=args.num_epoch, batch_size = args.batch_size \
                            ,num_workers =8\
                            ,note=args.data_size+'_'+args.data_type+'_'+args.train_set)
    model.execution(train_set,test_set)

