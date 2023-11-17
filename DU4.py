## Import libraries
import pandas as pd
import numpy as np
import scipy.integrate as integrate

## Ignore dumb warnings. We are adults. We take our chances
import warnings
warnings.filterwarnings('ignore')

## Add debug print statements
debug = False

## Define Crop Object
class Crop:
    '''
    Crop class
    '''
    def __init__(self, crop_type, filename_full='parameter-lists/crop_parameters_FPSD.xlsx'):
        '''
        Initializes the crop class
        :param crop_type: String that initializes the crop class. See the parameters file for more information
        :type crop_type: string
        :param filename_full: String that sets the filemane
        :type filename_full: String
        '''
        ## Create a pandas dataframe from the excel spreadsheet and convert to object dictionary
        df = pd.read_excel(filename_full, sheet_name='PxC')
        df = df.set_index('crop_type')
        temp_dict = df.loc[crop_type].to_dict()
        self.__dict__ =  temp_dict

## Define all the Modified Energy Cascade Equations
def calc_Phi_gammaE(crop, Phi_gamma):
    '''
    Calculates Effective photosynthetic photon flux 'Phi_gammaE' from Phi_gamma
    :param crop: crop object
    :type crop: object
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: effective Phi_gammaE in [umol_photon / (m^2 * s)]
    :rtype: float
    '''
    Phi_gammaE = Phi_gamma * (crop.H / crop.H_0)
    if debug:
        print('Phi_gammaE', Phi_gammaE)

    return Phi_gammaE


def calc_Y_Qmax(crop, c_CO2, Phi_gamma):
    '''
    Calculates the maximum canopy quantum yield Y_Qmax in [umol_C/umol_photon]
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return:  maximum canopy quantum yield Y_Qmax in [umol_C/umol_photon]
    :rtype: float
    '''
    Phi_gammas = [(1 / Phi_gamma), 1, Phi_gamma, Phi_gamma ** 2, Phi_gamma ** 3]
    c_CO2s = [(1 / c_CO2), 1, c_CO2, c_CO2 ** 2, c_CO2 ** 3]
    Y_Qmax = 0
    n = 0
    for i in range(0, len(Phi_gammas)):
        for j in range(0, len(c_CO2s)):
            C = eval('crop.C_' + str(n + 1))
            Y_Qmax = Y_Qmax + Phi_gammas[i] * c_CO2s[j] * C
            n = n + 1

    if debug:
        print('Y_Qmax', Y_Qmax)
        
    return Y_Qmax


def calc_t_A(crop, c_CO2, Phi_gamma):
    '''
    Calculates time until canopy closure t_A in days after emergence [d_AE]
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: time until canopy closure t_A in days after emergence [d_AE]
    :rtype: float
    '''
    Phi_gamma = calc_Phi_gammaE(crop, Phi_gamma)
    Phi_gammas = [(1 / Phi_gamma), 1, Phi_gamma, Phi_gamma ** 2, Phi_gamma ** 3]
    c_CO2s = [(1 / c_CO2), 1, c_CO2, c_CO2 ** 2, c_CO2 ** 3]
    t_A = 0
    n = 0
    for i in range(0, len(Phi_gammas)):
        for j in range(0, len(c_CO2s)):
            D = eval('crop.D_' + str(n + 1))
            t_A = t_A + Phi_gammas[i] * c_CO2s[j] * D
            n = n + 1

    if debug:
        print('t_A', t_A)

    return t_A


def calc_eta_C(t, crop):
    '''
    Calculates 24-hour carbon use efficency eta_C [fraction]
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :return: 24-hour carbon use efficency eta_C [fraction]
    :rtype: float
    '''
    eta_C = 0
    if t <= crop.t_Q:
        eta_C = crop.eta_Cmax
    elif t > crop.t_Q:
        # elif (t > crop.T_q) & (t < crop.T_m):
        eta_C = crop.eta_Cmax - (crop.eta_Cmax - crop.eta_Cmin) * (t - crop.t_Q) / (crop.t_M - crop.t_Q)

    if debug:
        print('eta_C', eta_C)

    return eta_C


def calc_A(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the fraction of PPF absorbed by the plant canopy [fraction]
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: fraction of PPF absorbed by the plant canopy A [fraction]
    :rtype: float
    '''
    t_A = calc_t_A(crop, c_CO2, Phi_gamma)
    if t < t_A:
        A = crop.A_max * (t / t_A) ** (crop.a)
    else:
        A = crop.A_max

    if debug:
        print('--params', t, t_A, t<t_A, crop.A_max, crop.a)
        print('A', A)

    return A


def calc_Y_Q(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the canopy quantum yield
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: canopy quantum yield
    :rtype: float
    '''
    Y_Qmax = calc_Y_Qmax(crop, c_CO2, Phi_gamma)
    if t <= crop.t_Q:
        Y_Q = Y_Qmax
    # elif (t > crop.t_q) & (t < crop.t_m):
    elif t > crop.t_Q:
        Y_Q = Y_Qmax - (Y_Qmax - crop.Y_Qmin) * (t - crop.t_Q) / (crop.t_M - crop.t_Q)

    if debug:
        print('Y_Q', Y_Q)

    return Y_Q


def calc_n_C(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the daily carbon gain [mol_Carbon/(m^2*d)]
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: the daily carbon gain [mol_Carbon/(m^2*d)
    :rtype: float
    '''
    A = calc_A(t, crop, c_CO2, Phi_gamma)
    Y_Q = calc_Y_Q(t, crop, c_CO2, Phi_gamma)
    eta_C = calc_eta_C(t, crop)
    n_C = 0.0036 * crop.H * eta_C * A * Y_Q * Phi_gamma

    if debug:
        print('n_C', n_C)

    return n_C


def calc_m_B(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the crop growth rate m_B in [g/m^2*d]
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: the crop growth rate m_B in [g/m^2*d]
    :rtype: float
    '''
    n_C = calc_n_C(t, crop, c_CO2, Phi_gamma)
    #12.011 is the molar mass of carbon
    m_B = (12.011 / crop.w_C) * n_C
    
    if debug:
        print('m_B', m_B)
    
    return m_B


def calc_n_O2(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the daily oxygen (O2) production of the crop in [mol_O2/(m^2*d)]
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: the daily oxygen (O2) production (n_O2) of the crop in [mol_O2/(m^2*d)]
    :rtype: float
    '''
    n_C = calc_n_C(t, crop, c_CO2, Phi_gamma)
    n_O2 = crop.Y_O2 * n_C
    
    if debug:
        print('n_O2', n_O2)
    
    return n_O2


def calc_V_trs(t, crop, c_CO2, Phi_gamma):
    '''
    Calculates the crop transpiration rate V_trs
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: crop transpiration rate V_trs
    :rtype: float
    '''
    A = calc_A(t, crop, c_CO2, Phi_gamma)
    eta_C = calc_eta_C(t, crop)
    Y_Q = calc_Y_Q(t, crop, c_CO2, Phi_gamma)
    #Maybe we should make all the environmental variables function params
    #Using Earth P_atm at sea level
    P_atm = 101.325
    rho_water = 998.23
    #relative humidity
    h_R = 0.7
    #Time of local sol in Earth hours per local day
    t_sol = 24
    p_sat = 0.611 * np.exp((17.4 * crop.T_L) / (crop.T_L + 239))
    Delta_p = p_sat * (1 - h_R)
    n_psgross = A * Y_Q * Phi_gamma
    '''
    It appears n_psnet will break if you try to simplify it this way. 
    Lesson: don't trust the robot simplifier. Test with numbers!
    n_psnet = ((crop.H * eta_C - crop.H + t_sol) / t_sol) * n_psgross
    '''
    n_psnet = ((t_sol - crop.H) / t_sol + (crop.H * eta_C) / t_sol) * n_psgross
    if crop.philetype == 'planophile':
        g_atm = 2.5
        g_sto = (1.717 * crop.T_L - 19.96 - 10.54 * Delta_p) / c_CO2 * n_psnet
    elif crop.philetype == 'erectophile':
        g_atm = 5.5
        g_sto = (0.1389 + 15.32 * h_R) / c_CO2 * n_psnet
    g_sfc = (g_atm * g_sto) / (g_atm + g_sto)
    #18.015 is molar mass of water
    V_trs = 3600 * crop.H * (18.015 / rho_water) * g_sfc * (Delta_p / P_atm)
    
    if debug:
        print('V_trs', V_trs)

    return V_trs


def calc_n_psnet(t, crop, c_CO2, Phi_gamma):
    '''
    As calc_V_trs but returns n_psnet
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param c_CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type c_CO2: float
    :param Phi_gamma: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type Phi_gamma: float
    :return: net photosynthesis rate P_net
    :rtype: float
    '''
    A = calc_A(t, crop, c_CO2, Phi_gamma)
    eta_C = calc_eta_C(t, crop)
    Y_Q = calc_Y_Q(t, crop, c_CO2, Phi_gamma)
    t_sol = 24
    n_psgross = A * Y_Q * Phi_gamma
    n_psnet = ((crop.H * eta_C - crop.H + t_sol) / t_sol) * n_psgross
    
    if debug:
        print(n_psnet)

    return n_psnet


def MEC_model(t, y, crop, CO2, PPF):
    '''
    Differential equation model. 
    :param t: time in days after emergence
    :type t: float
    :param y: model variable
    :type y: model variable
    :param crop: crop object
    :type crop: object
    :param CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type CO2: float
    :param PPF: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type PPF: float
    :return: dydt
    :rtype: array
    '''

    ## Set state variables
    # B = y[0]
    # O2 = y[1]
    # H2O = y[2]
    Neq = len(y)

    ## Prepare dydt array
    dydt = np.zeros((1, Neq))

    ## Define dydt
    dydt[0, 0] = calc_m_B(t, crop, CO2, PPF)
    dydt[0, 1] = calc_n_O2(t, crop, CO2, PPF)
    # dydt[0,2] = calc_transpiration(t,crop,CO2,PPF)
    dydt[0, 2] = calc_V_trs(t, crop, CO2, PPF) - calc_H2O_in(t, crop, CO2, PPF)
    # dydt[0,2] = -calc_H2O_in(t,crop,CO2,PPF) * crop.H2O_uptake

    # dydt[0,1] = calc_DOP(t,CO2,PPF,crop)

    return [np.transpose(dydt)]

def calc_H2O_in(t, crop, CO2, PPF):
    '''
    Calculates the water uptake by the crop
    :param t: time in days after emergence
    :type t: float
    :param crop: crop object
    :type crop: object
    :param CO2: atmospheric concentration of carbon dioxide in [umol_CO2/mol_air]
    :type CO2: float
    :param PPF: photosynthetic photon flux in [umol_photon / (m^2 * s)]
    :type PPF: float
    :return: water uptake
    :rtype: float
    '''
    
    m_mol_H2O = 18.01528
    # rho = 998.23
    H2O_in = crop.Y_O2 * calc_n_C(t, crop, CO2, PPF) * m_mol_H2O

    if debug:
        print('H2O_in', H2O_in)

    return H2O_in
