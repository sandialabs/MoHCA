# -*- coding: utf-8 -*-
import ISU_PINNbasedHCA

def iastate(input_csv_path, output_csv_path, model_save_filepath):
  ''' Execute ISU hosting capacity algorithm on in_path CSV with output written as CSV to out_path. '''
  ''' Besides the in_path and out_path, more setting information is needed for the code running. The information of the testing system is shown below.'''

  system_name = 'ST_model'
  total_bus_number = 46                                                     # total bus number
  node_list_for_HC = [k+1 for k in range(total_bus_number)]                 # selected bus for HC analysis
  model_retrain = 0                                                        # 1 for retraining; 0 for not training
  inverter_control_setting = 'watt'                                         # two setting mode: var prioirty and watt priority
  inverter_advanced_control = 1                                             # 0->'without control'  1->'constant power factor' 2->'constant reactive power' 3->'active power-reactive power' 4->'voltage-reactive power'
  ret_value = ISU_PINNbasedHCA.PINN_HC(system_name, input_csv_path, output_csv_path, model_save_filepath, total_bus_number, node_list_for_HC, model_retrain, inverter_advanced_control, inverter_control_setting)
  
  return ret_value


input_csv_path  = r'.\test_data\model_inputs\ST_input_meter_data.csv'  
output_csv_path = r'.\test_data\model_outputs\ST_0.95pf_meter_data.csv'  
model_path      = r'.\test_data\saved_model\ST_model.pt'
ret_value = iastate(input_csv_path, output_csv_path, model_path)