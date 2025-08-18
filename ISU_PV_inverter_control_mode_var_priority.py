# -*- coding: utf-8 -*-
"""
This is the smart inverter control mode with var priority 

"""
#%%
import numpy as np
import math
# import matplotlib.pyplot as plt

def line_circle(P2, Q2, P3, Q3, nameplate_apparent_power_rating):

    a = Q3**2 / (P3 - P2)**2 +1
    b = -2 * P2 * Q3**2 / (P3 - P2)**2
    c = Q3**2 / (P3 - P2)**2 * P2**2 - nameplate_apparent_power_rating**2
    x = (np.sqrt(b**2 - 4 * a *c) - b) / (2 * a)
    y = Q3 / (P3 - P2) * (x - P2)
    return x, y




def constant_pf_mode(category, variable, parameter=None):
    """
    Calculate reactive power Q based on category, active power, and parameters.
    
    :param category: 'A' or 'B'
    :param variable: Active power P
    :param parameter: List of parameters where parameter[0] is the nameplate apparent power,
                      parameter[1] is the power factor, and parameter[2] is optional based on category
    :return: Array of reactive power values [Q_inj, Q_abs, Q_inj_PF, Q_abs_PF, Q_inj_std, Q_abs_std]
    """
    if parameter is None:
        parameter = [4.2, 0.9]  # Default values if not provided
    nameplate_apparent_power_rating = parameter[0]
    power_factor = parameter[1]
    P = variable[0]
    
    if category == 'A' or category == 'B':
        nameplate_active_power_rating = parameter[2] if len(parameter) > 2 else None
        
        # Calculate Q_injection standard
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
        elif P >= 0.05 * nameplate_active_power_rating and P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating],
                                  [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
        elif P > 0.2 * nameplate_active_power_rating and P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
        else:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
        
        # Calculate Q_absorption standard differently for categories A and B
        if category == 'A':
            if P < 0.05 * nameplate_active_power_rating:
                Q_abs_std = 0
            elif P >= 0.05 * nameplate_active_power_rating and P <= 0.2 * nameplate_active_power_rating:
                Q_abs_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating],
                                      [-0.06 * nameplate_apparent_power_rating, -0.25 * nameplate_apparent_power_rating])
            elif P > 0.2 * nameplate_active_power_rating and P <= np.sqrt(1 - 0.25**2) * nameplate_apparent_power_rating:
                Q_abs_std = -0.25 * nameplate_apparent_power_rating
            else:
                Q_abs_std = -np.sqrt(nameplate_apparent_power_rating**2 - P**2)
        elif category == 'B':
            Q_abs_std = -Q_inj_std
    else:
        raise ValueError('Unknown category for constant power factor mode.')
    
    # Calculate Q based on power factor
    if category == 'A' or category == 'B':
        nameplate_active_power_rating = parameter[2] if len(parameter) > 2 else None
        # Calculate Q_injection and Q_absorbtion based on power factor
        P_max = nameplate_apparent_power_rating * power_factor      
        #P_max = nameplate_apparent_power_rating
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_PF = 0
            Q_abs_PF = 0
        elif P <= P_max:
            Q_inj_PF = P * math.tan(math.acos(power_factor))
            Q_abs_PF = -Q_inj_PF
        elif P <= nameplate_apparent_power_rating:
            #P = P_max
            Q_inj_PF = np.sqrt(nameplate_apparent_power_rating**2 - P**2 )
            Q_abs_PF = -Q_inj_PF
        else:
            raise ValueError('The active power should not exceed the apparent power.')
    else:
        raise ValueError('Unknown category for constant power factor mode.')
    
    # Determine final Q_inj and Q_abs
    Q_inj = Q_inj_PF
    Q_abs = Q_abs_PF
    
    return np.array([Q_inj, Q_abs, Q_inj_PF, Q_abs_PF, Q_inj_std, Q_abs_std]), np.array(P)


def active_reactive_mode(category, variable, parameter):
    nameplate_active_power_rating, nameplate_active_power_rating_s, P_min, P_min_s, nameplate_apparent_power_rating = parameter

    P   = variable[0]
    P1  = max(0.2 * nameplate_active_power_rating, P_min)
    P2  = 0.5 * nameplate_active_power_rating
    P3  = nameplate_active_power_rating
    P1s = min(0.2 * nameplate_active_power_rating_s, P_min_s)
    P2s = 0.5 * nameplate_active_power_rating_s
    P3s = nameplate_active_power_rating_s
    # Q of standard
    if category == 'A':
        # Q_injection standard
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
        
        # Q_absorption standard
        if P < 0.05 * nameplate_active_power_rating:
            Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_abs_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [-0.06 * nameplate_apparent_power_rating, -0.25 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.25**2) * nameplate_apparent_power_rating:
            Q_abs_std = -0.25 * nameplate_apparent_power_rating
        else:
            Q_abs_std = -np.sqrt(nameplate_apparent_power_rating**2 - P**2)

    elif category == 'B':
        ## ANSI C84.1 range A:0.95-1.05 rated voltage
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
            Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
            Q_abs_std = -Q_inj_std
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
            Q_abs_std = -Q_inj_std
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
            Q_abs_std = -Q_inj_std
    else:
        raise ValueError('Unknown category for active reactive mode.')
    
    # Q of active reactive power
    Q3s = 0.44 * nameplate_apparent_power_rating
    
    if category == 'A':
        Q3 = -0.25 * nameplate_apparent_power_rating
    elif category == 'B':
        Q3 = -0.44 * nameplate_apparent_power_rating
    else:
        raise ValueError('Unknown category for active reactive mode.')
    
    if P <= P3s:
        Q_inj_APRP = Q3s
    elif P <= P2s:
        Q_inj_APRP = np.interp(P, [P3s, P2s], [Q3s, 0])
        Q_abs_APRP = 0
    elif P<=P2:
        Q_inj_APRP = 0
        Q_abs_APRP = 0
    elif P<=P3:
        Q_inj_APRP = 0
        Q_abs_APRP = np.interp(P, [P2, P3], [0, Q3])
    elif P>P3:
        Q_inj_APRP = 0
        Q_abs_APRP = Q3
    else:
        raise ValueError('P should not exceed the apparent pwoer.')
        
    # Q of final Q_inj and Q_abs
    P_max = np.sqrt(nameplate_apparent_power_rating**2 - Q3**2)
    if P3 <= P_max:
        if P <= P3s:
            Q_inj = Q3s
            Q_abs = 0
        elif P <= P2s:
            Q_inj = np.interp(P, [P3s, P2s], [Q3s, 0])
            Q_abs = 0
        elif P<=P2:
            Q_inj = 0
            Q_abs = 0
        elif P<=P3:
            Q_inj = 0
            Q_abs = np.interp(P, [P2, P3], [0, Q3])
        elif P<=P_max:
            Q_inj = 0
            Q_abs = Q3
        elif P<=nameplate_apparent_power_rating:
            P = P_max
            Q_inj = 0
            Q_abs = Q3
        else:
            raise ValueError('P should not exceed the apparent pwoer.')
    elif P3 <= nameplate_apparent_power_rating:
        if P <= P3s:
            Q_inj = Q3s
            Q_abs = 0
        elif P <= P2s:
            Q_inj = np.interp(P, [P3s, P2s], [Q3s, 0])
            Q_abs = 0
        elif P<=P2:
            Q_inj = 0
            Q_abs = 0
        elif P<=P3:
            Q_inj = 0
            Q_abs = np.interp(P, [P2, P3], [0, Q3])
        elif P<=nameplate_apparent_power_rating:
            Q_inj = 0
            Q_abs = Q3
        else:
            raise ValueError('P should not exceed the apparent pwoer.')
        if P**2 + Q_abs**2 > nameplate_apparent_power_rating**2:
            P, Q_abs = line_circle(P2, 0, P3, Q3, nameplate_apparent_power_rating)
    else:
        raise ValueError('Prated should not exceed the apparent pwoer.')
    return np.array([Q_inj,Q_abs,Q_inj_APRP,Q_abs_APRP,Q_inj_std,Q_abs_std]), np.array(P)

def constant_reactive_mode(category, variable, *parameter):
    parameter = parameter[0]
    nameplate_apparent_power_rating = parameter[0]
    P = variable[0]
    nameplate_active_power_rating = parameter[1]  
    # Q of standard
    if category == 'A':
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
        
        if P < 0.05 * nameplate_active_power_rating:
            Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_abs_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [-0.06 * nameplate_apparent_power_rating, -0.25 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.25**2) * nameplate_apparent_power_rating:
            Q_abs_std = -0.25 * nameplate_apparent_power_rating
        elif P <= nameplate_apparent_power_rating:
            Q_abs_std = -np.sqrt(nameplate_apparent_power_rating**2 - P**2)
    elif category == 'B':
        nameplate_active_power_rating = parameter[2];
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
            Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
            Q_abs_std = -Q_inj_std
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
            Q_abs_std = -Q_inj_std
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
            Q_abs_std = -Q_inj_std
    else:
        raise ValueError("Unknown category for constant reactive mode.")

    # Q of constant reactive power
    Q_inj_RP = parameter[2]
    Q_abs_RP = parameter[3]
    # Q of output
    if P**2 + Q_inj_RP**2 > nameplate_apparent_power_rating**2:
        P_inj = np.sqrt(nameplate_apparent_power_rating**2 - Q_inj_RP**2)
    else:
        P_inj = P
    if P**2 + Q_abs_RP**2 > nameplate_apparent_power_rating**2:
        P_abs = np.sqrt(nameplate_apparent_power_rating**2 - Q_abs_RP**2)
    else:
        P_abs = P
    Q_inj = Q_inj_RP
    Q_abs = Q_abs_RP
    Q = [Q_inj, Q_abs, Q_inj_RP, Q_abs_RP, Q_inj_std, Q_abs_std]
    return Q, [P_inj, P_abs]


def vol_var_mode(category, variable, parameter):
    V_nominal, nameplate_apparent_power_rating, nameplate_active_power_rating = parameter[:3]
    # P = parameter[3]
    # V = variable
    P = variable[0]
    V = variable[1]

    # Q of standard
    Q_inj_std, Q_abs_std = 0, 0
    if category == 'A':
        # Q_injection 
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
     #  Q_absorbtion
        if P < 0.05 * nameplate_active_power_rating:
            Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_abs_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [-0.06 * nameplate_apparent_power_rating, -0.25 * nameplate_apparent_power_rating])
        elif P <= np.sqrt(1 - 0.25**2) * nameplate_apparent_power_rating:
            Q_abs_std = -0.25 * nameplate_apparent_power_rating
        elif P <= nameplate_apparent_power_rating:
            Q_abs_std = -np.sqrt(nameplate_apparent_power_rating**2 - P**2)   
        
    elif category == 'B':
        if P < 0.05 * nameplate_active_power_rating:
            Q_inj_std = Q_abs_std = 0
        elif P <= 0.2 * nameplate_active_power_rating:
            Q_inj_std = np.interp(P, [0.05 * nameplate_active_power_rating, 0.2 * nameplate_active_power_rating], [0.11 * nameplate_apparent_power_rating, 0.44 * nameplate_apparent_power_rating])
            Q_abs_std = -Q_inj_std
        elif P <= np.sqrt(1 - 0.44**2) * nameplate_apparent_power_rating:
            Q_inj_std = 0.44 * nameplate_apparent_power_rating
            Q_abs_std = -Q_inj_std
        elif P <= nameplate_apparent_power_rating:
            Q_inj_std = np.sqrt(nameplate_apparent_power_rating**2 - P**2)
            Q_abs_std = -Q_inj_std
    else:
        raise ValueError("Unknown category for vol var mode.")

    # Q of vol-var
    VRef_B = V_nominal;
    V1_A   = 0.9 * V_nominal
    V2_A   = V_nominal
    V3_A   = V_nominal
    V4_A   = 1.1 * V_nominal
    V1_B   = VRef_B - 0.08 * V_nominal
    V2_B   = VRef_B - 0.02 * V_nominal
    V3_B   = VRef_B + 0.02 * V_nominal
    V4_B   = VRef_B + 0.08 * V_nominal

    Q1_A = 0.25 * nameplate_apparent_power_rating # 25% of nameplate apparent power rating, injection
    Q4_A = 0.25 * nameplate_apparent_power_rating # 25% of nameplate apparent power rating, absorption
    Q1_B = 0.44 * nameplate_apparent_power_rating # 44% of nameplate apparent power rating, injection
    Q4_B = 0.44 * nameplate_apparent_power_rating # 44% of nameplate apparent power rating, absorption

    if category == 'A':
        V1 = V1_A
        V2 = V2_A
        V3 = V3_A
        V4 = V4_A
        Q1 = Q1_A
        Q2 = 0
        Q3 = 0
        Q4 = Q4_A
    elif category == 'B':
        V1 = V1_B
        V2 = V2_B
        V3 = V3_B
        V4 = V4_B
        Q1 = Q1_B
        Q2 = 0
        Q3 = 0
        Q4 = Q4_B
    else:
        raise ValueError("Unknown category for voltage-reactive power mode.")
        
    # Q_VV
    VL = 0.8
    VH = 1.2
    Q_VV = 0
    if V <= VL:  
        raise ValueError("The lowest voltage is 0.8.")
    elif V <= V1:
        Q_VV = Q1
    elif V <= V2:
        Q_VV = np.interp(V, [V1, V2], [Q1, Q2])
    elif V <= V3:
        Q_VV = 0
    elif V <= V4:
        Q_VV = -np.interp(V, [V3, V4], [Q3, Q4])
    elif V <= VH:  
        Q_VV = -Q4
    else:
        raise ValueError("The highest voltage is 1.2.")
    
    # Final Q
    if P**2 + Q_VV**2 > nameplate_apparent_power_rating**2:
        P = np.sqrt(nameplate_apparent_power_rating**2 - Q_VV**2)
    Q = Q_VV


    return [Q, Q_inj_std, Q_abs_std, Q_VV], P

def active_power_voltage_mode(category, variable, parameter):
    Pmin, nameplate_active_power_rating, VN = parameter[:3]
    V = variable[0]
    PV_Gen_P = variable[1]

    #P of active power voltage, default setting
    V1 = 1.06 * VN
    V2 = 1.1  * VN
    VH = 1.2  * VN
    P2 = min(0.2 * nameplate_active_power_rating,Pmin);
    Vmin= 0.8 * VN;

        
    # Q_VV
    VL = 0.8
    VH = 1.2
    Q_VV = 0
    if V <= VL:  
        raise ValueError("The lowest voltage is 0.8.")
    elif V <= V1:
        P = nameplate_active_power_rating
    elif V <= V2:
        P = np.interp(V, [V1, V2], [nameplate_active_power_rating, P2])
    elif V <= VH:  
        P = P2 
    else:
        raise ValueError("The highest voltage is 1.2.")
    return min(P,PV_Gen_P)



def control_mode(modes, category, variable, *parameters):
    """
    Control mode function to select and execute different reactive power control modes.
    
    Parameters:
    - modes: The control mode to be executed ('constant power factor', 'voltage-reactive power', 
             'active power-reactive power', 'constant reactive power')
    - category: The category type (e.g., 'A' or 'B')
    - variable: The primary variable for the control mode (e.g., active power, voltage)
    - parameters: Parameters required by the specific control mode function
    
    Returns:
    - Q: The output from the selected control mode function, typically reactive power values.
    """
    if modes == 'constant power factor':
        # Ensure the constant_pf_mode function is defined
        Q, P = constant_pf_mode(category, variable, *parameters)
    elif modes == 'voltage-reactive power':
        # Ensure the vol_var_mode function is defined
        Q, P = vol_var_mode(category, variable, *parameters)
    elif modes == 'active power-reactive power':
        # Ensure the active_reactive_mode function is defined
        Q, P = active_reactive_mode(category, variable, *parameters)
    elif modes == 'constant reactive power':
        # Ensure the constant_reactive_mode function is defined
        Q, P = constant_reactive_mode(category, variable, *parameters)
    elif modes == 'active power voltage':
        # Ensure the active_power_voltage function is defined
        P = active_power_voltage_mode(category, variable, *parameters)
        Q = 0
    else:
        raise ValueError('Unknown mode selected.')
    return Q, P

def reactive_power_generation(mode_control, PV_Gen_P, voltage, PV_installed):
    """
    Calculate reactive power generation Q_gen based on control mode, active power,
    voltage, and installed PV capacity.

    Args:
    mode_control (str): Control mode.
    PV_Gen_P (float): Active power generated by the PV system.
    voltage (float): Voltage and P.
    PV_installed (float): Installed capacity of the PV system.

    Returns:
    float: Reactive power generation Q_gen.
    """
    Q_gen = None

    if mode_control == 'without control':
        Q_gen = 0
        P_gen = PV_Gen_P

    elif mode_control == 'constant power factor':
        
        power_factor = -0.95
        
        if PV_Gen_P < 0.05 * PV_installed:
            PV_Gen_Q = [0, 0, 0, 0, 0, 0]
            PV_Gen_P_actual = PV_Gen_P
        else:
            PV_nameplate = PV_installed
            # Assume control_mode is defined elsewhere
            PV_Gen_Q, PV_Gen_P_actual = control_mode('constant power factor', 'A', [PV_Gen_P, voltage], [PV_nameplate, power_factor, PV_installed])
        if power_factor>0:
            Q_gen = PV_Gen_Q[1]
        else:
            Q_gen = PV_Gen_Q[0]
        P_gen = PV_Gen_P_actual


    elif mode_control == 'voltage-reactive power':
        if PV_Gen_P < 0.05 * PV_installed:
            PV_Gen_Q = [0, 0, 0, 0, 0, 0]
            PV_Gen_P_actual = PV_Gen_P
        else:        
            V_nominal = 1
            PV_nameplate = PV_installed * 1.0
            nameplate_apparent_power_rating = PV_nameplate
            nameplate_active_power_rating = PV_installed
            parameter = [V_nominal, nameplate_apparent_power_rating, nameplate_active_power_rating, PV_Gen_P]
            # Assume control_mode is defined elsewhere
            PV_Gen_Q, PV_Gen_P_actual = control_mode('voltage-reactive power', 'A', [PV_Gen_P, voltage], parameter)
        Q_gen = PV_Gen_Q[0]
        P_gen = PV_Gen_P_actual

    elif mode_control == 'active power-reactive power':
        nameplate_active_power_rating = PV_installed
        nameplate_active_power_rating_s = -PV_installed
        P_min = 0
        P_min_s = 0
        nameplate_apparent_power_rating = PV_installed * 1.0

        if PV_Gen_P < 0.05 * PV_installed:
            PV_Gen_Q = [0, 0, 0, 0, 0, 0]
            PV_Gen_P_actual = PV_Gen_P
        else:
            # Assume control_mode is defined elsewhere
            PV_Gen_Q, PV_Gen_P_actual = control_mode('active power-reactive power', 'A', [PV_Gen_P, voltage], [nameplate_active_power_rating, nameplate_active_power_rating_s, P_min, P_min_s, nameplate_apparent_power_rating])
        Q_gen = PV_Gen_Q[1]
        P_gen = PV_Gen_P_actual

    elif mode_control == 'constant reactive power':
        if PV_Gen_P < 0.05 * PV_installed:
            PV_Gen_Q = [0, 0, 0, 0, 0, 0]
            PV_Gen_P_actual = [PV_Gen_P,PV_Gen_P]
        else:            
            nameplate_apparent_power_rating = PV_installed * 1.0
            nameplate_active_power = PV_installed
            injection_reactive_power = 0.125 * PV_installed
            absorption_reactive_power = -0.125 * PV_installed
    
            parameter = [nameplate_apparent_power_rating, nameplate_active_power, injection_reactive_power, absorption_reactive_power]
            # Assume control_mode is defined elsewhere
            PV_Gen_Q, PV_Gen_P_actual = control_mode('constant reactive power', 'B', [PV_Gen_P, voltage]
                                    , parameter)
        Q_gen = PV_Gen_Q[1]
        
        P_gen = PV_Gen_P_actual[0]
    elif mode_control == 'active power voltage':
        if PV_Gen_P < 0.05 * PV_installed:
            Q_gen = 0
            PV_Gen_P_actual = PV_Gen_P
        else:           
            
            V_nominal = 1
            nameplate_apparent_power_rating = PV_installed * 1.0
            nameplate_active_power_rating = PV_installed
            Pmin = 0.05 * nameplate_apparent_power_rating

            parameter = [Pmin,nameplate_active_power_rating,V_nominal]
            Q_gen, PV_Gen_P = control_mode('active power voltage', [],[voltage, PV_Gen_P], parameter)
            
        P_gen = PV_Gen_P
        Q_gen = 0
        

    else:
        raise ValueError('Unknown mode selected.')

    return Q_gen, P_gen

#%%
#Q_gen, P_gen_actual = reactive_power_generation('constant power factor', 0.8, 9999, 1)
#print(Q_gen)


