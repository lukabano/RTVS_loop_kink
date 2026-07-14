# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:04:20 2025

@author: lukab
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ------------ Units and constants --------------------------------------

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
a = 1.0 + 4.0*He_abundance
b = 2.0 + 3.0*He_abundance
# derived units
unit_density = a*m_p*unit_numberdensity 
unit_pressure = b*unit_numberdensity*k_B*unit_temperature
unit_velocity = (unit_pressure/unit_density)**0.5
unit_time = unit_length/unit_velocity
unit_heat = unit_pressure/unit_time
#magnetic field is in units where miu0 = 1
unit_magneticfield = (mu_0*unit_pressure)**0.5


# ------------ User input --------------------------------------

L = 10.0e9 /unit_length #10.0e9/unit_length # half-length of the loop (i.e. bottom to top) in cm

g_0 = 2.74e4 *unit_length/unit_velocity**2   # cm/s^2
kappa = 1.7e-6 *unit_temperature**3.5/unit_length/unit_density/unit_velocity**3  # Spitzer conductivity in cgs 

# rad. losses approximated as E_R = chi * T^alpha
alpha = -0.5
chi = 10**(-18.8) / unit_pressure*unit_numberdensity**2 * unit_time*unit_temperature**alpha * (1+2*He_abundance) # cgs; last factor from Hermans&Keppens21

r_0 = 1.e8/unit_length      # loop radius at the footpoint in cm
zeta_0 = 5                  # density contrast at the footpoint           
B_0 = 20./unit_magneticfield # magnetic field at the footpoint in G
f = 0.1                     # filling factor

# Main part of the input

# footpoint pressure in dyn/cm^2
P_0 = 0.0659e0 / unit_pressure

# footpoint temperature in K corresponding to the top of the chromosphere
# shouldn't be changed (much) to keep the physical assumptions behind the model valid
T_0 = 2e4 / unit_temperature 

# initial guess for max. temperature (at the loop apex)
# should be set higher than the anticipated value
T_max = 3e6 / unit_temperature  # initial guess

# initial guess for kink wave energy density at the footpoint (at the loop apex)
# should be set much higher than the anticipated value
W_k_0 = 1.67e2 / unit_pressure  # erg/cm^3 = 0.1 J/m^3

# Example of a high-temp input

# P_0 = 0.5e1 / unit_pressure   # dyn/cm^2  # initial guess:1
# T_0 = 2e4 / unit_temperature
# T_max = 5e6 / unit_temperature  # initial guess
# W_k_0 = 8e2 / unit_pressure  # erg/cm^3 = 0.1 J/m^3


# footpoint distance from the photosphere in cm
s_left = 1e8/unit_length # 2.104e8/unit_length

# initial number of gridpoints; the code will refine the grid automatically if necessary
n_points = 100

# maximum allowed number of gridpoints
n_points_max = 100000


# ------------ Solver --------------------------------------

#%% In terms of variables Q = ln(P/P_0), eta = T**7/2 and W = 1/sqrt(WW), where WW = alpha_kink * B * W_k
# extra equation for W_k for both directions

s = np.linspace(s_left, L+s_left, num=n_points)

y = np.empty((5, s.size))
y[0] = np.zeros((1, s.size))
y[1] = np.linspace(T_0**3.5, T_max**3.5, n_points)
#y[2] = np.full((1, s.size), (T_max-T_0)/(L/n_points))
y[2] = np.linspace((T_max**3.5-T_0**3.5)/(L/n_points), 0., n_points)

W_0 = (W_k_0 * B_0 * np.sqrt(2/(1+1./zeta_0) * T_0 / P_0))**(-0.5)
y[3] = W_0 * s/s_left

y[4] = y[3, -1] * (s[-1]+s_left-s)/s_left

#%% Restarting using the last solution as the initial condition

# s = solution.x

# y = np.empty((5, s.size))

# y[0] = solution.y[0]
# y[1] = solution.y[1]
# y[2] = solution.y[2]
# y[3] = solution.y[3]
# y[4] = solution.y[4]


#%%

def gradient(quantity, x):
    '''Calculates the gradient of a scalar field quantity in all the cells except the two on the domain edge'''
    
    size = len(quantity)
    
    #cell_faces = np.concat((np.full(1, s[0]-(s[1]-s[0])/2), s[:len(s)-1]+(s[1:]-s[:len(s)-1])/2, np.full(1, s[-1]+(s[-1]-s[-2])/2)))
    #cell_faces = s[:len(s)-1]+(s[1:]-s[:len(s)-1])/2
    #cell_sizes = cell_faces[1:]-cell_faces[:len(s)]
    
    return (quantity[2:size]-quantity[0:size-2])/(x[2:size]-x[0:size-2])

def equations(s, y, param):
    Q = y[0]
    eta = y[1]
    diffeta = y[2]
    W = y[3]
    W_plus = y[4]
    
    #B = B_0 * pow(1.+ L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-2)
    B = B_0 * pow(1.+ 2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-2)
    diffB = 1./unit_length * (-B_0/R_sun) * pow(1.+ L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-3) * np.cos(np.pi*(s-s_left)/(2*L))
    #diffB = (-B_0/R_sun) * pow(1.+ L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-3) * np.cos(np.pi*(s-s_left)/(2*L))
    
    zeta = (zeta_0-1)*np.exp(-2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5)+1
    #zeta = (zeta_0-1)*np.exp(-L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5)+1
    diffzeta = -1./unit_length * (zeta_0-1)*np.exp(-L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5) / 10/R_sun * np.cos(np.pi*(s-s_left)/(2*L))
    #diffzeta = - (zeta_0-1)*np.exp(-L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5) / 10/R_sun * np.cos(np.pi*(s-s_left)/(2*L))
    
    r = np.sqrt(B_0/B)*r_0
    L_perp = pow(zeta+1-f,3/2.)/(1-pow(f,5/2.))/(zeta-1)*np.sqrt(10)*np.sqrt(f*np.pi)*r
    alpha_kink = np.sqrt(2/zeta * eta**(2./7) / (P_0*np.exp(Q)) * (1-f+f*zeta))
    W_k = W**(-2) / alpha_kink / B
    W_k_plus = W_plus**(-2) / alpha_kink / B
    
    E_H = np.sqrt(eta**(2./7) * (1-f+f*zeta) / P_0 / np.exp(Q)) * (W_k**1.5 + W_k_plus**(1.5)) / L_perp
    rad_loss = (1-f+f*zeta**2) / (1-f+f*zeta)**2 * chi * P_0**2 * np.exp(2*Q) * eta**(2./7*alpha-4./7)
    
    W_derivative_expression = 1./L_perp/B**1.5 * (P_0*np.exp(Q)/eta**(2./7))**0.25 * zeta**0.75 / 2**1.75 / (1-f+f*zeta)
    W_plus_derivative_expression = -1./L_perp/B**1.5 * (P_0*np.exp(Q)/eta**(2./7))**0.25 * zeta**0.75 / 2**1.75 / (1-f+f*zeta)
    
    #W_k_derivative_expression = -2 * W_derivative_expression * W_k**1.5
    #W_k_plus_derivative_expression = -2 * W_plus_derivative_expression * W_k_plus**1.5
    
    kink_pressure = (1+zeta)/4 * (W_k+W_k_plus)
    #diffkink_pressure = 1./4*diffzeta*(W_k+W_k_plus) + (1+zeta)/4*(W_k_derivative_expression + W_k_plus_derivative_expression)
    
    R = 2*f*diffzeta/(1-f+f*zeta) - diffzeta/zeta + 2./7*diffeta/eta
    diffkink_pressure_red = 1./4*diffzeta*(W_k+W_k_plus) - (1+zeta)/4*(2*W_derivative_expression/W * W_k + 2*W_plus_derivative_expression/W_plus * W_k_plus + (W_k+W_k_plus)*(R/2 + diffB/B))
    
    # diffkink_pressure = np.empty((s.size))
    # #print(kink_pressure[0])
    
    # h1 = s[1]-s[0]
    # h2 = s[2]-s[1]
    # diffkink_pressure[0] = (-(2*h1+h2)/(h1*(h1+h2)))*kink_pressure[0] + ((h1+h2)/(h1*h2))*kink_pressure[1] + (-h1/(h2*(h1+h2)))*kink_pressure[2]
    
    # h1 = s[-2]-s[-1]
    # h2 = s[-3]-s[-2]
    # diffkink_pressure[-1] = (-(2*h1+h2)/(h1*(h1+h2)))*kink_pressure[-1] + ((h1+h2)/(h1*h2))*kink_pressure[-2] + (-h1/(h2*(h1+h2)))*kink_pressure[-3]
    
    # plt.plot(s, diffkink_pressure)
    # plt.show()
    
    # for i in range(1, len(kink_pressure)-1):
    #     diffkink_pressure[i] = (kink_pressure[i+1] - kink_pressure[i-1]) / (s[i+1]-s[i-1])
    
    
    # deprecated
    #Q_derivative = -P_0 * g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7)
    #Q_derivative = -P_0 * g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7) + np.exp(Q) * B*diffB
    #Q_derivative = (-P_0 * g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7) + np.exp(Q) * (B*diffB - diffkink_pressure_red)) / (1+(1+zeta)/8 * (W_k + W_k_plus))
    
    # correct
    Q_derivative = - g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7)
    #Q_derivative = - g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7) + B*diffB / (P_0*np.exp(Q))
    #Q_derivative = (- g_0 * np.cos(np.pi/2 * (s-s_left)/L) * eta**(-2./7) + 1./(P_0*np.exp(Q)) * (B*diffB - diffkink_pressure_red)) / (1+(1+zeta)/8 * (W_k + W_k_plus))
    
    eta_derivative = diffeta
    diffeta_derivative = 3.5/kappa * (-E_H + rad_loss)
    #WW_derivative = -1./L_perp/B**1.5 * (P_0*np.exp(Q)/eta**(2./7))**0.25 * ((zeta+1)/2)**0.75 / zeta**1.25 * WW**1.5
    W_derivative = W_derivative_expression
    W_plus_derivative = W_plus_derivative_expression
    
    return np.vstack((Q_derivative, eta_derivative, diffeta_derivative, W_derivative, W_plus_derivative))

def boundary_conditions(left, right, param):
    T_max = param[0]
    # E_0 = param[1]
    W_0 = param[1]
    return np.array([left[0], left[1]-T_0**3.5, right[1]-T_max**3.5, left[2], right[2], left[3]-W_0, right[3]-right[4]])

# actual solver
solution = solve_bvp(equations, boundary_conditions, s, y, p=[T_max, W_0], max_nodes=n_points_max, verbose=2)


# control printouts
print('T_max [MK] = ', solution.p[0])
print('W_0 = ', solution.p[1])
# print('E_0 [erg cm**-3 s**-1] = ', solution.p[1]*unit_pressure/unit_time)
# print('W_k [erg cm**-3] = ', solution.p[1]*unit_pressure)

Q = solution.y[0]
eta = solution.y[1]
diffeta = solution.y[2]
W = solution.y[3]
W_plus = solution.y[4]
s = solution.x

# ------------ Plotting --------------------------------------

# Calculating the relevant quantities derived from the solution
B = B_0 * pow(1.+ 2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) / R_sun,-2)
zeta = (zeta_0-1)*np.exp(-2*L/np.pi * np.sin(np.pi*(s-s_left)/(2*L)) /R_sun/5)+1
r = np.sqrt(B_0/B)*r_0
L_perp = pow(zeta+1-f,3/2.)/(1-pow(f,5/2.))/(zeta-1)*np.sqrt(10)*np.sqrt(f*np.pi)*r
alpha_kink = np.sqrt(2/zeta * eta**(2./7) / (P_0*np.exp(Q)) * (1-f+f*zeta))
W_k = W**(-2) / alpha_kink / B
W_k_plus = W_plus**(-2) / alpha_kink / B
P_k = (1+zeta)/4 * (W_k+W_k_plus)
E_H = np.sqrt(eta**(2./7) * (1-f+f*zeta) / P_0 / np.exp(Q)) * (W_k**1.5 + W_k_plus**(1.5)) / L_perp
rad_loss = (1-f+f*zeta**2) / (1-f+f*zeta)**2 * chi * P_0**2 * np.exp(2*Q) * eta**(2./7*alpha-4./7)

temp = eta**(2./7)
conductive_flux = kappa * temp[1:len(s)-1]**2.5 * gradient(temp, s)
thermal_conduction = gradient(conductive_flux, s)
E_total = E_H[2:len(s)-2] + thermal_conduction - rad_loss[2:len(s)-2]

average_W_k = ((W_k*unit_pressure)**1.5 + (W_k_plus*unit_pressure)**1.5)**(2./3)
density = P_0 * np.exp(Q) / (eta**(2./7)) * unit_density

# User input: specify variables to plot

var_to_plot = [np.exp(Q), np.exp(Q)*P_0*unit_pressure, eta**(2./7), diffeta, W, W_plus, W_k*unit_pressure, W_k_plus*unit_pressure, W_k*unit_pressure + W_k_plus*unit_pressure, average_W_k, E_H*unit_pressure/unit_time, np.log10(density)]
title_to_plot = ["$P/P_0$", "$P$ [dyn/cm$^2$]", "$T$ [MK]", "$\\eta'$", "$W$", "$W^+$", "$W_k$ [erg/cm$^3$]", "$W_k^+$ [erg/cm$^3$]", "total $W_k$ [erg/cm$^3$]", 'average $W_k$ [erg/cm$^3$]', '$E_H$ [erg/cm$^3/s$]', '$\\log(\\rho$ [g/cm$^3$])']

# Actual plotting

for i in range(len(var_to_plot)):
    plt.plot(s, var_to_plot[i])
    #plt.xlim(0.01*L+s_left, L+s_left)
    plt.xlim(s_left, L+s_left)
    plt.title(title_to_plot[i])
    plt.show()

# Additional plots

plt.plot(s, rad_loss * unit_pressure/unit_time)
plt.plot(s, E_H * unit_pressure/unit_time)
plt.plot(s[2:len(s)-2], thermal_conduction * unit_pressure/unit_time)
plt.plot(s[2:len(s)-2], E_total * unit_pressure/unit_time, color='k')
plt.xlim(0.0*L+s_left, L+s_left)
#plt.yscale('log')
plt.ylim(-3e-5, 8e-4)
plt.hlines(0, np.min(s), np.max(s),  lw=1, color='k', ls=':')
plt.title("$E_R$ or $E_H$ [erg cm$^{-3}$ s$^{-1}$]")
plt.show()

#%% Custom plots

# W_k over the whole loop

s_whole_loop = np.concat((s, 2*(s_left+L)-np.flip(s)))
W_k_whole_loop = np.concat((W_k, np.flip(W_k_plus)))
plotting_indices = np.intersect1d(np.where(0.01*L+s_left <= s_whole_loop)[0], np.where(s_whole_loop <= 1.99*L+s_left)[0])

plt.plot(s_whole_loop, W_k_whole_loop*unit_pressure)
plt.xlim(0.01*L+s_left, 1.99*L+s_left)
plt.ylim(np.min(W_k_whole_loop[plotting_indices]*unit_pressure), np.max(W_k_whole_loop[plotting_indices]*unit_pressure))
plt.title("$W_k$ [erg/cm$^3$]")
plt.show()


# ------------ Saving the data --------------------------------------


#data_file = './P10_L50Mm_Wkwhole.txt'
#np.savetxt(data_file, np.column_stack((s_whole_loop, W_k_whole_loop*unit_pressure)), header='s W_k_whole_loop')

#%% Saving data for R file, Jupyter notebook

# data_file = './P10_L50Mm.txt'
# np.savetxt(data_file, np.column_stack((s, eta, np.exp(Q)*P_0*unit_pressure, average_W_k, E_H*unit_pressure/unit_time)), header='s eta P W_k_total E_H')


#%% Saving data for plotting with UAWSoM: s in Mm, Te in MK, the rest in cgs units

#data_file = './for_UAWSoM/RTVs_16m_scaledchi_L100Mm.txt'
#np.savetxt(data_file, np.column_stack((s-s_left, eta**(2./7), density, np.exp(Q)*P_0*unit_pressure, W_k*unit_pressure, W_k_plus*unit_pressure, rad_loss * unit_pressure/unit_time, E_H*unit_pressure/unit_time)), header='s Te rho p wkminus wkplus rad Qk')

#%% Saving data for Haruka

#data_file = './Haruka/T9500_zeta05_f010.txt'
#np.savetxt(data_file, np.column_stack((s, eta**(2./7), np.exp(Q)*P_0*unit_pressure, P_k*unit_pressure, W_k*unit_pressure, W_k_plus*unit_pressure, E_H*unit_pressure/unit_time)), header='s[Mm] T[MK] P[dyn/cm^2] P_k[dyn/cm^2] W_k^-[erg/cm^3] W_k^+[erg/cm^3] E_H[erg/cm^3/s]')

#%% Saving data for Max

data_file = './Max/T8000_2L107_test.txt'
np.savetxt(data_file, np.column_stack((s, eta**(2./7), density, np.exp(Q)*P_0*unit_pressure, P_k*unit_pressure, W_k*unit_pressure, W_k_plus*unit_pressure, E_H*unit_pressure/unit_time)), header='s[Mm] T[MK] rho[g/cm^3] P[dyn/cm^2] P_k[dyn/cm^2] W_k^-[erg/cm^3] W_k^+[erg/cm^3] E_H[erg/cm^3/s]')
