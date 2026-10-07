#%% ---- 1: INITIAL DECLARATIONS -------------------------------------------

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

#system of units (relative to CGS): 
# user specified units
unit_length = 1.e8 # cm
unit_numberdensity = 1.e9 # cm^-3
unit_temperature = 1.e6 # K

# constants
He_abundance = 0.1 
m_p = 1.67e-24 # g
k_B =  1.3806488e-16 # erg K^-1
mu_0 = 4*np.pi
R_sun = 6.957e10/unit_length # cm
mu = 1.0 + 4.0*He_abundance
nu = 2.0 + 3.0*He_abundance
# derived units
unit_density = mu*m_p*unit_numberdensity 
unit_pressure = nu*unit_numberdensity*k_B*unit_temperature
unit_velocity = (unit_pressure/unit_density)**0.5
unit_time = unit_length/unit_velocity
unit_heat = unit_pressure/unit_time
# magnetic field is in units where mu_0 = 1
unit_magneticfield = (mu_0*unit_pressure)**0.5


#%% ---- 2: USER INPUT PARAMETERS -------------------------------------------
# physical units refer to the values before normalisation is applied

g_0 = 2.74e4 *unit_length/unit_velocity**2   # gravitational acceleration at the footpoint in cm/s^2
kappa = 1.7e-6 *unit_temperature**3.5/unit_length/unit_density/unit_velocity**3   # Spitzer conductivity in erg cm^-1 s^-1 K^-7/2
alpha = -0.5                                 # radiative cooling exponent
chi = 10**(-18.8) / unit_pressure*unit_numberdensity**2 * unit_time*unit_temperature**alpha * (1+2*He_abundance) # radiative cooling coefficient; last factor from Hermans&Keppens21

r_0 = 1.e8/unit_length          # loop radius at the footpoint in cm
zeta_0 = 5                      # density contrast at the footpoint
B_0 = 20./unit_magneticfield    # magnetic field strength at the footpoint in G
f = 0.1                         # filling factor

L = 10.e9 /unit_length              # loop half-length (bottom to top) in cm
P_0 = 0.6e0 / unit_pressure         # base pressure in dyn/cm^2
T_0 = 2.0e4 / unit_temperature      # temperature at the base in K

# initial guesses
T_max = 3e6 / unit_temperature      # initial guess for max. temperature in K
                                    # should be set higher than the anticipated value
W_k_0 = 1.67e2 / unit_pressure      # initial guess for W_k^- at the footpoint in erg/cm^3 = 0.1 J/m^3
                                    # should be set significantly higher than the anticipated value (by at least an order of magnitude)

# mesh setup
n_points = 100              # initial no. of cells; the mesh is refined dynamically by the solver where needed
max_n_points = 10000        # max. no of cells that the solver is allowed to refine to
s_left = 1e8/unit_length    # distance of the loop footpoint to the solar surface in cm

# for saving the data
save_folder = './public/'
save_file_name = 'P06_L100'


#%% ---- 3: SOLVER -------------------------------------------

# Equations solved in terms of variables 
# Q = ln(P/P_0) (\Tilde{P} in Banović+26), 
# eta = T**7/2 (\Tilde{T} in Banović+26) and 
# W^\pm = 1/sqrt(WW^\pm) (\Tilde{S} in Banović+26), where WW^\pm = alpha_kink * B * W_k^\pm


# Initialisation of variables

s = np.linspace(s_left, L+s_left, num=n_points)

y = np.empty((5, s.size))
y[0] = np.zeros((1, s.size))
y[1] = np.linspace(T_0**3.5, T_max**3.5, n_points)
y[2] = np.linspace((T_max**3.5-T_0**3.5)/(L/n_points), 0., n_points)

W_0 = (W_k_0 * B_0 * np.sqrt(2/(1+1./zeta_0) * T_0 / P_0))**(-0.5)
y[3] = W_0 * s/s_left

y[4] = y[3, -1] * (s[-1]+s_left-s)/s_left


# definitions of functions needed for the solver

def gradient(quantity, x):
    '''Calculates the gradient of a scalar field accurate to order 2. quantity and x have to be of the same length'''
    
    size = len(quantity)    
    grad = np.zeros_like(quantity)
    
    h1 = x[1]-x[0]
    h2 = x[2]-x[1]
    grad[0] = -(2*h1+h2)/h1/(h1+h2) * quantity[0] + (h1+h2)/h1/h2 * quantity[1] - h1/h2/(h1+h2) * quantity[2]
    
    hplus = x[2:size]-x[1:size-1]
    hminus = x[1:size-1]-x[0:size-2]
    cminus = -hplus/hminus / (hplus+hminus)
    c0 = (hplus-hminus) / hplus/hminus
    cplus = hminus/hplus / (hplus+hminus)
    grad[1:size-1] = cminus * quantity[0:size-2] + c0 * quantity[1:size-1] + cplus * quantity[2:size]
    
    h1 = x[size-2]-x[size-1]
    h2 = x[size-3]-x[size-2]
    grad[size-1] = -(2*h1+h2)/h1/(h1+h2) * quantity[size-1] + (h1+h2)/h1/h2 * quantity[size-2] - h1/h2/(h1+h2) * quantity[size-3]
        
    return grad


def equations(s, y, param):
    Q = y[0]
    eta = y[1]
    diffeta = y[2]
    W_min = y[3]
    W_plus = y[4]
    
    B = B_0 * pow(1.+ 2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-2)    
    zeta = (zeta_0-1)*np.exp(-2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5)+1   
    r = np.sqrt(B_0/B)*r_0
    L_perp = pow(zeta+1-f,3/2.)/(1-pow(f,5/2.))/(zeta-1)*np.sqrt(10)*np.sqrt(f*np.pi)*r
    alpha_kink = np.sqrt(2/zeta * eta**(2./7) / (P_0*np.exp(Q)) * (1-f+f*zeta))
    
    W_k_min = W_min**(-2) / alpha_kink / B
    W_k_plus = W_plus**(-2) / alpha_kink / B
    
    E_H = np.sqrt(eta**(2./7) * (1-f+f*zeta) / P_0 / np.exp(Q)) * (W_k_min**1.5 + W_k_plus**(1.5)) / L_perp
    E_R = (1-f+f*zeta**2) / (1-f+f*zeta)**2 * chi * P_0**2 * np.exp(2*Q) * eta**(2./7*alpha-4./7)
    
    W_derivative_expression = 1./L_perp/B**1.5 * (P_0*np.exp(Q)/eta**(2./7))**0.25 * zeta**0.75 / 2**1.75 / (1-f+f*zeta)
    W_plus_derivative_expression = -1./L_perp/B**1.5 * (P_0*np.exp(Q)/eta**(2./7))**0.25 * zeta**0.75 / 2**1.75 / (1-f+f*zeta)
    

    Q_derivative = - g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7)
    
    eta_derivative = diffeta
    diffeta_derivative = 3.5/kappa * (-E_H + E_R)
    W_derivative = W_derivative_expression
    W_plus_derivative = W_plus_derivative_expression
    
    return np.vstack((Q_derivative, eta_derivative, diffeta_derivative, W_derivative, W_plus_derivative))

def boundary_conditions(left, right, param):
    T_max = param[0]
    W_0 = param[1]
    return np.array([left[0], left[1]-T_0**3.5, right[1]-T_max**3.5, left[2], right[2], left[3]-W_0, right[3]-right[4]])

# call to the solver
solution = solve_bvp(equations, boundary_conditions, s, y, p=[T_max, W_0], max_nodes=max_n_points, verbose=2)

print('T_max [MK] = ', solution.p[0])
print('W_0 = ', solution.p[1])

#%% ---- 4: PLOTTING -------------------------------------------

# recover all the necessary quantities from the solution

Q = solution.y[0]
eta = solution.y[1]
diffeta = solution.y[2]
W_min = solution.y[3]
W_plus = solution.y[4]
s = solution.x

B = B_0 * pow(1.+ 2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-2)
zeta = (zeta_0-1)*np.exp(-2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5)+1
r = np.sqrt(B_0/B)*r_0
L_perp = pow(zeta+1-f,3/2.)/(1-pow(f,5/2.))/(zeta-1)*np.sqrt(10)*np.sqrt(f*np.pi)*r
alpha_kink = np.sqrt(2/zeta * eta**(2./7) / (P_0*np.exp(Q)) * (1-f+f*zeta))

W_k_min = W_min**(-2) / alpha_kink / B
W_k_plus = W_plus**(-2) / alpha_kink / B

P_k = (1+zeta)/4 * (W_k_min+W_k_plus)
E_H = np.sqrt(eta**(2./7) * (1-f+f*zeta) / P_0 / np.exp(Q)) * (W_k_min**1.5 + W_k_plus**(1.5)) / L_perp
E_R = (1-f+f*zeta**2) / (1-f+f*zeta)**2 * chi * P_0**2 * np.exp(2*Q) * eta**(2./7*alpha-4./7)

temp = eta**(2./7)
conductive_flux = kappa * temp**2.5 * gradient(temp, s)
thermal_conduction = gradient(conductive_flux, s)
E_total = E_H + thermal_conduction - E_R

average_W_k = ((W_k_min*unit_pressure)**1.5 + (W_k_plus*unit_pressure)**1.5)**(2./3)  # \bar{W}_k from Banović+26
density = P_0 * np.exp(Q) / (eta**(2./7)) * unit_density


# separate plots for a variety of quantities

var_to_plot = [np.exp(Q), np.exp(Q)*P_0*unit_pressure, eta**(2./7), diffeta, W_min, W_plus, W_k_min*unit_pressure, W_k_plus*unit_pressure, W_k_min*unit_pressure + W_k_plus*unit_pressure, average_W_k, E_H*unit_pressure/unit_time, np.log10(density)]
title_to_plot = ["$P/P_0$", "$P$ [dyn/cm$^2$]", "$T$ [MK]", "$\\eta'$", "$W^-$", "$W^+$", "$W_k^-$ [erg/cm$^3$]", "$W_k^+$ [erg/cm$^3$]", "$W_k^-+W_k^+$ [erg/cm$^3$]", '$\\bar{W}_k$ [erg/cm$^3$]', '$E_H$ [erg/cm$^3/s$]', '$\\rho$ [g/cm$^3$]']

for i in range(len(var_to_plot)):
    plt.plot(s, var_to_plot[i])
    plt.xlim(s_left, L+s_left)
    plt.title(title_to_plot[i])
    plt.show()

    
# joint plot for all the energy components

plt.plot(s, E_R * unit_pressure/unit_time, label='$E_R$')
plt.plot(s, E_H * unit_pressure/unit_time, label='$E_H$')
plt.plot(s, thermal_conduction * unit_pressure/unit_time, label='$F_c$')
plt.plot(s, E_total * unit_pressure/unit_time, color='k', label='total')
plt.xlim(0.0*L+s_left, L+s_left)
#plt.yscale('log')
plt.ylim(-8e-4, 8e-3)
plt.hlines(0, np.min(s), np.max(s),  lw=1, color='k', ls=':')
plt.legend()
plt.title("energy rates [erg cm$^{-3}$ s$^{-1}$]")
plt.show()


# plot of W_k_min over the whole loop

s_whole_loop = np.concat((s, 2*(s_left+L)-np.flip(s)))
W_k_min_whole_loop = np.concat((W_k_min, np.flip(W_k_plus)))
plotting_indices = np.intersect1d(np.where(0.01*L+s_left <= s_whole_loop)[0], np.where(s_whole_loop <= 1.99*L+s_left)[0])

plt.plot(s_whole_loop, W_k_min_whole_loop*unit_pressure)
plt.xlim(0.01*L+s_left, 1.99*L+s_left)
plt.ylim(np.min(W_k_min_whole_loop[plotting_indices]*unit_pressure), np.max(W_k_min_whole_loop[plotting_indices]*unit_pressure))
plt.title("$W_k^-$ [erg/cm$^3$]")
plt.show()

#%% ---- 5: SAVING THE DATA -------------------------------------------

data_file = save_folder + save_file_name + '.txt'
np.savetxt(data_file, np.column_stack((s, eta**(2./7), density, np.exp(Q)*P_0*unit_pressure, P_k*unit_pressure, W_k_min*unit_pressure, W_k_plus*unit_pressure, E_R * unit_pressure/unit_time, E_H*unit_pressure/unit_time)), header='s[Mm] T[MK] rho[g/cm^3] P[dyn/cm^2] P_k[dyn/cm^2] W_k^-[erg/cm^3] W_k^+[erg/cm^3] E_R[erg/cm^3/s] E_H[erg/cm^3/s]')
