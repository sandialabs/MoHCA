# -*- coding: utf-8 -*-
"""
This is the draft version of PINN_based PV hosting capacity analysis codes.

"""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import pandas as pd
from torch.autograd import Variable
import os
import random
import time
import sys
import warnings

warnings.filterwarnings("ignore")


def PINN_HC(testing_system_name:str, input_csv_path:str, output_csv_path:str, node_number:int, nodes_selected=0, retrain_indicator=0, inverter_control=0, control_setting='var'):
    """

    Parameters
    ----------
    testing_system_name : str
        the name of the testing system.
    input_csv_path : str
        the file path of the input data.
    output_csv_path : str
        the file path of the output result data.
    node_number : int
        the customer number.
    nodes_selected : list, optional
        the selected nodes to calculate LHC. The default is 0, whhich means all the nodes are analyzed.
    retrain_indicator: int
        1 represents the model needs to be retrain; 0 denotes the model has been trained
    inverter_control: int
        0: without any control; 1:constant power factor; 2: constant var; 3: watt-var; 4: volt-var
    control_setting: str
        var: var-priority mode; watt: watt-priority mode

    Returns
    -------
    the LHC results.

    """
    
    
    
    def set_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)    
        return device

    def set_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        
    device = set_device()
    set_seed()

    def TrainTestSplit(x_data, y_data, perc, node_number):  
        device = set_device()

        splitpoint = round(x_data.shape[0]*perc)
        x_data_train = torch.tensor(x_data[:splitpoint :], dtype=torch.float32).to(device)
        y_data_train = torch.tensor(y_data[:splitpoint :], dtype=torch.float32).to(device)
        x_data_test = torch.tensor(x_data[splitpoint: :], dtype=torch.float32).to(device)
        y_data_test = torch.tensor(y_data[splitpoint: :], dtype=torch.float32).to(device) 
        
        x_data_train_P_xf = (x_data_train[:,:node_number] - x_data_train[:,:node_number].mean())/ x_data_train[:,:node_number].std()
        x_data_train_Q_xf = (x_data_train[:,node_number:] - x_data_train[:,node_number:].mean())/ x_data_train[:,node_number:].std()
        
        y_data_train_xf = (y_data_train - y_data_train.mean())/ y_data_train.std()  
        x_data_train_xf = torch.concat((x_data_train_P_xf, x_data_train_Q_xf), axis=1)

        
        x_data_test_P_xf = (x_data_test[:,:node_number] - x_data_train[:,:node_number].mean())/ x_data_train[:,:node_number].std()
        x_data_test_Q_xf = (x_data_test[:,node_number:] - x_data_train[:,node_number:].mean())/ x_data_train[:,node_number:].std()
        
        x_data_test_xf = torch.concat((x_data_test_P_xf, x_data_test_Q_xf), axis=1)

        x_reverse_P_xf_mean = torch.ones(node_number, dtype=torch.float32).to(device)
        x_reverse_P_xf_std = torch.ones(node_number, dtype=torch.float32).to(device)
        x_reverse_P_xf_mean = x_reverse_P_xf_mean*x_data_train[:,:node_number].mean()
        x_reverse_P_xf_std = x_reverse_P_xf_std*x_data_train[:,:node_number].std() 

        x_reverse_Q_xf_mean = torch.ones(node_number, dtype=torch.float32).to(device)
        x_reverse_Q_xf_std = torch.ones(node_number, dtype=torch.float32).to(device)
        x_reverse_Q_xf_mean = x_reverse_Q_xf_mean*x_data_train[:,node_number:].mean()
        x_reverse_Q_xf_std = x_reverse_Q_xf_std*x_data_train[:,node_number:].std() 
        

        x_reverse_xf_mean = torch.concat((x_reverse_P_xf_mean, x_reverse_Q_xf_mean))
        x_reverse_xf_std = torch.concat((x_reverse_P_xf_std, x_reverse_Q_xf_std))

        
        y_reverse_xf_mean = torch.ones(node_number, dtype=torch.float32).to(device)
        y_reverse_xf_std = torch.ones(node_number, dtype=torch.float32).to(device)
        

        y_reverse_xf_mean = y_reverse_xf_mean*y_data_train[:,:node_number].mean()
        y_reverse_xf_std = y_reverse_xf_std*y_data_train[:,:node_number].std() 
        
        return x_data_train_xf, \
            y_data_train_xf, \
            x_data_test_xf, \
            y_data_test, \
            y_reverse_xf_mean,\
            y_reverse_xf_std,\
            x_reverse_xf_mean,\
            x_reverse_xf_std 
            

    
    def model_selection(model_name, node_number):
        '''
        Parameters
        ----------
        model_name : char
            Select specific model for the voltage calculation
        node_number : int
            Enter the nodenumber as the customer number
        Returns
        -------
        Return initialized model

        '''
        class Linearlize_totalpower_OLTC_Net(nn.Module):

            def __init__(self, node_number):
                super(Linearlize_totalpower_OLTC_Net, self).__init__()
    
                self.node_number = node_number
                self.A = nn.Linear(node_number, node_number, bias=False)
                self.A.weight = torch.nn.Parameter(-torch.from_numpy(np.identity(node_number)))# initialize the weight of B layer
                self.B = nn.Linear(node_number, node_number, bias=False)
                self.B.weight = torch.nn.Parameter(-torch.from_numpy(np.identity(node_number)))# initialize the weight for test
                self.K =  Variable(torch.randn(1, node_number).type(dtype=torch.float32), requires_grad=True).to(device)
    
                self.Layer1 = nn.Linear(3*node_number, 2*node_number, bias=True)
                self.Layer1.weight = torch.nn.Parameter(torch.from_numpy(np.zeros( (2*node_number, node_number*3) )))
                self.Layer2 = nn.Linear(2*node_number, node_number, bias=True)
                self.Layer2.weight = torch.nn.Parameter(torch.from_numpy(np.zeros( (node_number, node_number*2))))
    
                self.AF1 = nn.Sigmoid()
                self.AF2 = nn.LeakyReLU(0.2)
                self.AF3 = nn.ReLU()
                self.AF4 = nn.Tanh()
    
    
                self.Total_load_adjust1 = nn.Linear(2*node_number + 2, 2*node_number, bias=True)
                self.Total_load_adjust2 = nn.Linear(2*node_number, node_number, bias=True)
                self.Total_load_adjust1.weight = torch.nn.Parameter(torch.from_numpy(np.zeros( (2*node_number , 2*node_number +2))))
                self.Total_load_adjust2.weight = torch.nn.Parameter(torch.from_numpy(np.zeros( (node_number, 2*node_number))))
    
    
                self.Combine = nn.Linear(node_number, node_number, bias=True)
                self.Combine.weight = torch.nn.Parameter(torch.from_numpy( np.identity(node_number)  ) )# initialize the weight for test
                # self.Combine.bias = torch.nn.Parameter(torch.from_numpy(  np.random.rand(node_number).reshape((node_number,1))   ) )# initialize the weight for test
    
                self.dropout = nn.Dropout(0.25)
    
                for name, p in self.Combine.named_parameters():
                    if name=='weight':
                        p.requires_grad=False
    
            def forward(self, x):
                xp = self.A(x[:,:self.node_number])
                xq = self.B(x[:,self.node_number:])
    
                xpq = xp + xq
    
                v = self.Combine(xpq)
    
                x_2 = torch.pow(x, 2)
                xpqv = torch.concat((x_2, v), dim=1)
                xpqv = self.Layer1(xpqv)
                xpqv = self.AF4(xpqv)
                xpqv = self.Layer2(xpqv)
                xpqv = self.AF4(xpqv)
    
                v_final = xpqv + v
                xp_sum = torch.reshape(torch.sum(xp, axis = 1), (-1,1))
                xq_sum = torch.reshape(torch.sum(xq, axis = 1), (-1,1))
    
                total_pqv = torch.concat((xp_sum, xq_sum, xp, xq), dim=1)
                total_pqv = self.Total_load_adjust1(total_pqv)
                total_pqv = self.AF4(total_pqv)
                total_pqv = self.Total_load_adjust2(total_pqv)
                total_pqv = self.AF4(total_pqv)
    
                total_pqv_final = v_final + total_pqv
    
    
                return total_pqv_final, xpqv, total_pqv,v
    
    
        model = Linearlize_totalpower_OLTC_Net(node_number).float().to(device)
        return model
    
    # ********************************************************************************
    #                     Data loading and formating
    # ********************************************************************************

    data = pd.read_csv(input_csv_path)
    
    P = np.array(data['kw_reading']).reshape(node_number,-1).T
    Q = np.array(data['kvar_reading']).reshape(node_number,-1).T
    PQ = np.hstack((P, Q))


    V = np.array(data['v_reading']).reshape(node_number,-1).T
    
    
    V_base = []
    for i in range(node_number):
        if len(np.where(V[:,i]<220)[0])>3000:
            v_base = 208
        else:
            v_base = 240
        V_base.append(v_base)

    V = V / V_base
    
    pd.DataFrame(PQ).to_csv('PQ_EC2_tttt.csv', header=None)
    pd.DataFrame(V).to_csv('V_EC2_tttt.csv', header=None)
    
    #V = V / v_base # convert to Per Unit Value
    V = np.power(V, 2)
    
    
    V_data = V
    PQ_data = PQ
    
    if retrain_indicator==0:
        batch_size = P.shape[0]
    else:
        batch_size = 1500 # batch size for model training   
    
    x_data_train_xf, y_data_train_xf, x_data_test_xf, y_data_test_xf, y_reverse_xf_mean, y_reverse_xf_std, x_reverse_xf_mean,x_reverse_xf_std = TrainTestSplit(PQ_data, V_data, 0.95, node_number)
    torch_dataset_train = Data.TensorDataset(x_data_train_xf, y_data_train_xf)
    
    loader_train = Data.DataLoader( 
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  #
        )
    
    # ********************************************************************************
    #                     Model Building an Training
    # ********************************************************************************
    
    model = model_selection('Linearlize_totalpower_Net', node_number)
    
    if retrain_indicator==0:
        
        try:
            model_name = testing_system_name + '.pt'
            #print(model_name)
            model.load_state_dict(torch.load(model_name))
        except:
            
            print('Cannot find the model...')
            print('Please correct system name or retrain model...')
            sys.exit()

        model_name = testing_system_name + '.pt'
        model.load_state_dict(torch.load(model_name))
   
        
        actual_test_results = []
        predict_test_results = []
        with torch.no_grad():
            MAE = 0
            for step, (batch_x, batch_y) in enumerate(loader_train):
                batch_x_data = batch_x.to(device)
                target = batch_y.to(device)
                predict,xpqv, total_pqv,v = model(batch_x_data)
                target = target*y_reverse_xf_std + y_reverse_xf_mean
                actual_test_results = actual_test_results + target.reshape(-1).tolist()
                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                
                predict_test_results = predict_test_results + predict.reshape(-1).tolist()
                MAE = MAE + torch.mean(abs(target - predict))

            
            
            voltage_matrix_actual= np.power(np.array(actual_test_results).reshape(-1, node_number), 0.5)
            voltage_matrix_predeict= np.power(np.array(predict_test_results).reshape(-1, node_number), 0.5)
            
            MAE_customer = abs(voltage_matrix_actual - voltage_matrix_predeict).mean()
            
            
            if MAE_customer>0.0005:
                print(MAE_customer)
                print('Model is not accurate, please retrain the model...')
                sys.exit()
             
        
    else:

        loss_func = nn.MSELoss()
        epoch_parameter = 10000
        Beta = 0.00005
        regularization_type = 'L'
        #------------Adaptive Learning Rate---------------
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.02)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:0.95**(epoch//500))
        Epoch_loss_record = []
    
        
        for i in range(epoch_parameter):
            running_loss = 0.0
            running_regular = 0.0
            loss_view = 0.0
            print("| Learning Rate in Epoch {0} ： {1:4.2f}|".format(i, optimizer.param_groups[0]['lr']))
        
            for step, (batch_x, batch_y) in enumerate(loader_train):
        
                out, linearlize_error_comp, uplevel_voltage_influence, v= model(batch_x)
        
        
                if regularization_type == 'L':
                    # A weight
                    A_weight = torch.norm(model.A.weight, 2)
                    A_w = torch.norm(model.A.weight, 1)
                    # B weight
                    B_weight = torch.norm(model.B.weight, 2)
                    B_w = torch.norm(model.B.weight, 1)
                    # linearlize_error_comp
                    error_comp = torch.norm(linearlize_error_comp, 2)
                    # uplevel_voltage_influence
                    voltage_influence = torch.norm(uplevel_voltage_influence, 2)
                    # keep A and B positive
        
                    keep_positive = torch.abs(-model.A.weight.sum() - torch.norm(model.A.weight, 1)) + torch.abs(-model.B.weight.sum() - torch.norm(model.B.weight, 1))
        
                    regularization_loss = Beta*( A_weight + 0.00*A_w + B_weight + 0.00*B_w + 0.1*error_comp + 0*voltage_influence + 0.1*keep_positive)
                    # regularization_loss = Beta*( A_weight + B_weight + 0.1*error_comp ++ 0.1*keep_positive)
                    regularization_loss = regularization_loss.cpu()
        
                elif regularization_type == 'No':
        
                    regularization_loss = 0
        
        
                loss = loss_func(out,batch_y) + regularization_loss + 0.2*loss_func(v,batch_y)
                optimizer.zero_grad()
                loss.backward()
        
                temp_A = (model.A.weight.grad.clone() + model.A.weight.grad.clone().T)/2
                model.A.weight.grad = nn.Parameter(temp_A)
                temp_B = (model.B.weight.grad.clone() + model.B.weight.grad.clone().T)/2
                model.B.weight.grad = nn.Parameter(temp_B)
        
                optimizer.step()
                time = 0
                running_loss += loss.item()
                running_regular = running_regular + regularization_loss
                loss_view += loss.item()
        
            scheduler.step()
        
        
            print('| Epoch:{} | Sum_Loss:{:.5f} | Reg:{:5.2f} | Loss:{:5.2f} |'.format(i+1, loss_view, running_regular, loss_view-running_regular))
            Epoch_loss_record.append(loss_view)
        
        
        
        actual_test_results = []
        predict_test_results = []
        with torch.no_grad():
            MAE = 0
            for step, (batch_x, batch_y) in enumerate(loader_train):
                batch_x_data = batch_x.to(device)
                target = batch_y.to(device)
                predict,xpqv, total_pqv,v = model(batch_x_data)
                target = target*y_reverse_xf_std + y_reverse_xf_mean
                actual_test_results = actual_test_results + target.reshape(-1).tolist()
                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                predict_test_results = predict_test_results + predict.reshape(-1).tolist()
                MAE = MAE + torch.mean(abs(target - predict))

            
            voltage_matrix_actual= np.power(np.array(actual_test_results).reshape(-1, node_number), 0.5)
            voltage_matrix_predeict= np.power(np.array(predict_test_results).reshape(-1, node_number), 0.5)
            
            np.save('Ck5_actual.npy', voltage_matrix_actual)
            np.save('Ck5_predict.npy', voltage_matrix_predeict)
            
            MAE_customer = abs(voltage_matrix_actual - voltage_matrix_predeict).mean()
            
            if MAE_customer>0.001:
                print('Model is not accurate, please retrain the model...')
                sys.exit()

        
        filepath = testing_system_name + '.pt'
        torch.save(model.state_dict(), filepath)  
    

    # ********************************************************************************
    #                     Locational Hosting Capacity Analysis
    # ********************************************************************************
    # 1Day 1Node Hosting Capacity
    
    data['datetime'] = pd.to_datetime(data['datetime'])
    filtered_data = data[(data['datetime'].dt.hour >= 9) & (data['datetime'].dt.hour < 14)]
    P = np.array(filtered_data['kw_reading']).reshape(node_number,-1).T
    Q = np.array(filtered_data['kvar_reading']).reshape(node_number,-1).T
    PQ = np.hstack((P, Q))

    V_base = []
    for i in range(node_number):
        if len(np.where(V[:,i]<220)[0])>2000:
            v_base = 208
        else:
            v_base = 240
        V_base.append(v_base)    
    
    V = np.array(filtered_data['v_reading']).reshape(node_number,-1).T
    V = V / v_base # convert to Per Unit Value
    V = np.power(V, 2)
    V_data = V
    PQ_data = PQ
    
    
    
    
    PQ_extreme_all = PQ_data
    V_extreme_all = V_data
    PQ_extreme_record = PQ_extreme_all.copy()
    
    #Irrandiance = np.array(pd.read_csv('Irrandiance.csv',index_col=None)).reshape(-1)
    Irrandiance = np.ones(V_extreme_all.shape[0])
    
    
    
    if nodes_selected == 0:
        selectednodes = [i for i in range(node_number)]
    else:
        selectednodes = nodes_selected
    
    
    PV_capacity_record_all_nodes = pd.DataFrame({'busname':[], 'kW_hostable': []})
    
    if control_setting == 'var':
        from ISU_PV_inverter_control_mode_var_priority import reactive_power_generation
    elif control_setting == 'watt':
        from ISU_PV_inverter_control_mode_watt_priority import reactive_power_generation
    else:
        print('Wrong control setting...')
        raise SystemExit
        
        
    if inverter_control==0:
        control_name = 'without control' 
    elif inverter_control==1:
        control_name = 'constant power factor' 
    elif inverter_control==2:
        control_name = 'constant reactive power' 
    elif inverter_control==3:
        control_name = 'active power-reactive power'
    elif inverter_control==4:
        control_name = 'voltage-reactive power' 
    else:
        print('Wrong control mode...')
        raise SystemExit
    print(control_name)
        


    PQ_extreme_record = PQ_extreme_all.copy()
    V_extreme_record  = V_extreme_all.copy()
    PV_capacity_record_all_nodes = []
    PV_capacity_record_all_nodes = pd.DataFrame({'busname':[], 'kW_hostable': []})
    
    
    y_reverse_xf_std = y_reverse_xf_std.to(device)
    y_reverse_xf_mean = y_reverse_xf_mean.to(device)
    
    for node in selectednodes:
        print('/========= Analysing Bus '+ str(node) + ' ...' +'=========/')
        V_temp = np.ones_like(V_extreme_record)
        P_install = 0
        select_node = node + 1
        V_constrain = 0
        PQ_extreme = PQ_extreme_all.copy()
     

        while V_constrain < 0.05:
            detlaP = 30 
            Last_itera_record = PQ_extreme.copy()
            P_target = PQ_extreme[:,select_node-1]
            P_target = P_target - detlaP*Irrandiance
            P_install = P_install + detlaP
            PQ_extreme[:,select_node-1] = P_target
              
            PQ_extreme_temp = PQ_extreme_record.copy()
            PQ_extreme_temp[:,select_node-1] = PQ_extreme_record[:,select_node-1] - P_install*Irrandiance
            PQ_extreme_valt_var = PQ_extreme_temp.copy()
          
            if (control_name=='voltage-reactive power') or (control_name=='active power voltage'):
                V_record_temp = np.zeros(P_target.size)
                iteration = 0           
                while iteration<30:
                    Q_convergence = []
                    iteration = iteration + 1
                    x_data_test_extreme = torch.tensor(PQ_extreme_valt_var, dtype=torch.float32).to(device)
                    x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)

                    with torch.no_grad():
                        predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                        predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                        predict = predict.view(-1).detach().cpu().numpy()
                        
                        V_temp = np.power(np.array(predict),0.5).reshape(-1,node_number)

                    for i in range(P_target.size): #for i in range(P_target.size):
                        Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                        PQ_extreme_valt_var[i,select_node-1 + node_number] = PQ_extreme_temp[i,select_node-1 + node_number] - Q_gen
                        PQ_extreme_valt_var[i,select_node-1] = PQ_extreme_temp[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
                        Q_convergence.append(Q_gen)

                    if max(abs(V_temp[:,select_node-1]-V_record_temp))<0.001:
                        break
        
                    V_record_temp = V_temp[:,select_node-1]
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
    
            else:
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)

            x_data_test_extreme = torch.tensor(PQ_extreme, dtype=torch.float32).to(device)
            x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)
            with torch.no_grad(): 
                predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                predict = predict.view(-1).detach().cpu().numpy()
                V_constrain = np.sqrt(predict.max())-1
        
        P_install = P_install - detlaP

        V_constrain = 0
        PQ_extreme = Last_itera_record
        while V_constrain < 0.05:
            detlaP = 10
            Last_itera_record = PQ_extreme.copy()
            P_target = PQ_extreme[:,select_node-1]
            P_target = P_target - detlaP*Irrandiance
            P_install = P_install + detlaP
            PQ_extreme[:,select_node-1] = P_target
              
            PQ_extreme_temp = PQ_extreme_record.copy()
            PQ_extreme_temp[:,select_node-1] = PQ_extreme_record[:,select_node-1] - P_install*Irrandiance
            PQ_extreme_valt_var = PQ_extreme_temp.copy()
          
            if (control_name=='voltage-reactive power') or (control_name=='active power voltage'):
                V_record_temp = np.zeros(P_target.size)
                iteration = 0           
                while iteration<30:
                    Q_convergence = []
                    iteration = iteration + 1
                    x_data_test_extreme = torch.tensor(PQ_extreme_valt_var, dtype=torch.float32).to(device)
                    x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)

                    with torch.no_grad():
                        predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                        predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                        predict = predict.view(-1).detach().cpu().numpy()
                        V_temp = np.power(np.array(predict),0.5).reshape(-1,node_number)

                    for i in range(P_target.size): #for i in range(P_target.size):
                        Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                        PQ_extreme_valt_var[i,select_node-1 + node_number] = PQ_extreme_temp[i,select_node-1 + node_number] - Q_gen
                        PQ_extreme_valt_var[i,select_node-1] = PQ_extreme_temp[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
                        Q_convergence.append(Q_gen)

                    if max(abs(V_temp[:,select_node-1]-V_record_temp))<0.001:
                        break
        
                    V_record_temp = V_temp[:,select_node-1]
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
    
            else:
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)

            x_data_test_extreme = torch.tensor(PQ_extreme, dtype=torch.float32).to(device)
            x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)
            with torch.no_grad(): 
                predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                predict = predict.view(-1).detach().cpu().numpy()
                V_constrain = np.sqrt(predict.max())-1

        P_install = P_install - detlaP

        V_constrain = 0
        PQ_extreme = Last_itera_record
        while V_constrain < 0.05:
            detlaP = 2
            Last_itera_record = PQ_extreme.copy()
            P_target = PQ_extreme[:,select_node-1]
            P_target = P_target - detlaP*Irrandiance
            P_install = P_install + detlaP
            PQ_extreme[:,select_node-1] = P_target
              
            PQ_extreme_temp = PQ_extreme_record.copy()
            PQ_extreme_temp[:,select_node-1] = PQ_extreme_record[:,select_node-1] - P_install*Irrandiance
            PQ_extreme_valt_var = PQ_extreme_temp.copy()
          
            if (control_name=='voltage-reactive power') or (control_name=='active power voltage'):
                V_record_temp = np.zeros(P_target.size)
                iteration = 0           
                while iteration<30:
                    Q_convergence = []
                    iteration = iteration + 1
                    x_data_test_extreme = torch.tensor(PQ_extreme_valt_var, dtype=torch.float32).to(device)
                    x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)

                    with torch.no_grad():
                        predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                        predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                        predict = predict.view(-1).detach().cpu().numpy()
                        V_temp = np.power(np.array(predict),0.5).reshape(-1,node_number)

                    for i in range(P_target.size): #for i in range(P_target.size):
                        Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                        PQ_extreme_valt_var[i,select_node-1 + node_number] = PQ_extreme_temp[i,select_node-1 + node_number] - Q_gen
                        PQ_extreme_valt_var[i,select_node-1] = PQ_extreme_temp[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
                        Q_convergence.append(Q_gen)

                    if max(abs(V_temp[:,select_node-1]-V_record_temp))<0.001:
                        break
        
                    V_record_temp = V_temp[:,select_node-1]
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
    
            else:
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)

            x_data_test_extreme = torch.tensor(PQ_extreme, dtype=torch.float32).to(device)
            x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)
            with torch.no_grad(): 
                predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                predict = predict.view(-1).detach().cpu().numpy()
                V_constrain = np.sqrt(predict.max())-1
        
        P_install = P_install - detlaP
        V_constrain = 0
        PQ_extreme = Last_itera_record
        while V_constrain < 0.05:
            detlaP = 0.1
            Last_itera_record = PQ_extreme.copy()
            P_target = PQ_extreme[:,select_node-1]
            P_target = P_target - detlaP*Irrandiance
            P_install = P_install + detlaP
            PQ_extreme[:,select_node-1] = P_target
              
            PQ_extreme_temp = PQ_extreme_record.copy()
            PQ_extreme_temp[:,select_node-1] = PQ_extreme_record[:,select_node-1] - P_install*Irrandiance
            PQ_extreme_valt_var = PQ_extreme_temp.copy()
          
            if (control_name=='voltage-reactive power') or (control_name=='active power voltage'):
                V_record_temp = np.zeros(P_target.size)
                iteration = 0           
                while iteration<30:
                    Q_convergence = []
                    iteration = iteration + 1
                    x_data_test_extreme = torch.tensor(PQ_extreme_valt_var, dtype=torch.float32).to(device)
                    x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)

                    with torch.no_grad():
                        predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                        predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                        predict = predict.view(-1).detach().cpu().numpy()
                        V_temp = np.power(np.array(predict),0.5).reshape(-1,node_number)

                    for i in range(P_target.size): #for i in range(P_target.size):
                        Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                        PQ_extreme_valt_var[i,select_node-1 + node_number] = PQ_extreme_temp[i,select_node-1 + node_number] - Q_gen
                        PQ_extreme_valt_var[i,select_node-1] = PQ_extreme_temp[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
                        Q_convergence.append(Q_gen)

                    if max(abs(V_temp[:,select_node-1]-V_record_temp))<0.001:
                        break
        
                    V_record_temp = V_temp[:,select_node-1]
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)
    
            else:
                
                for i in range(P_target.size): 
                    Q_gen, P_gen_actual = reactive_power_generation(control_name, P_install*Irrandiance[i], V_temp[i,select_node-1], P_install)
                    
                    PQ_extreme[i,select_node-1 + node_number] = PQ_extreme_all[i,select_node-1 + node_number] - Q_gen
                    PQ_extreme[i,select_node-1] = PQ_extreme[i,select_node-1] + (P_install*Irrandiance[i]-P_gen_actual)

            x_data_test_extreme = torch.tensor(PQ_extreme, dtype=torch.float32).to(device)
            x_data_test_xf_extreme = (x_data_test_extreme - x_reverse_xf_mean.to(device)) / x_reverse_xf_std.to(device)
            with torch.no_grad(): 
                predict,xpqv, total_pqv, v= model(x_data_test_xf_extreme)                
                predict = predict*y_reverse_xf_std + y_reverse_xf_mean
                predict = predict.view(-1).detach().cpu().numpy()
                V_constrain = np.sqrt(predict.max())-1

        P_install = P_install - detlaP
        current_node_temp = pd.DataFrame({'busname':['bus'+str(node+1)], 'kW_hostable': P_install})
        PV_capacity_record_all_nodes = pd.concat([PV_capacity_record_all_nodes, current_node_temp]).reset_index(drop=True)

    PV_capacity_record_all_nodes.to_csv(output_csv_path, index=None)
    
    return PV_capacity_record_all_nodes


