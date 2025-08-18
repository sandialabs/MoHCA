# -*- coding: utf-8 -*-
import ISU_PINNbasedHCA

def iastate(input_csv_path, output_csv_path):
  ''' Execute ISU hosting capacity algorithm on in_path CSV with output written as CSV to out_path. '''
  ''' Besides the in_path and out_path, more setting information is needed for the code running. The information of the testing system is shown below.'''

  system_name = 'ST_model'
  node_list_for_HC = [1,2,3]                   # selected bus for HC analysis
  total_bus_number = 46                        # total bus number
  model_retrain = 1                            # 1 for retraining; 0 for not training
  inverter_control_setting = 'watt'            # two setting mode: var prioirty and watt priority
  inverter_advanced_control = 1                # 0->'without control'  1->'constant power factor' 2->'constant reactive power' 3->'active power-reactive power' 4->'voltage-reactive power'
  ret_value = ISU_PINNbasedHCA.PINN_HC(system_name, input_csv_path, output_csv_path, total_bus_number, nodes_selected=node_list_for_HC, retrain_indicator=model_retrain, inverter_control=inverter_advanced_control, control_setting=inverter_control_setting)
  
  return ret_value


input_csv_path = '\test_data\ST_Example_Files\ST_input_meter_data.csv'  
output_csv_path = '\test_data\ST_Example_Files\output\ST_output_0.95_pf.csv'     
ret_value = iastate(input_csv_path, output_csv_path)