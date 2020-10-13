'''
linear_fitting_functions.py

author: Sarah Wesolowski [scwesolowski@salisbury.edu]
last modified: Oct-8-2020
description: contains functions for math 465 students to perform different 
linear fits visually using a variety of criteria

'''

import numpy as np 
from scipy.optimize import linprog
from scipy.optimize import minimize #also could use their actual least sq function
import matplotlib.pyplot as plt

def visual_fit_linear(x_in, data_in, m_in, b_in, xpred_in):
    # get some information from stuff passed in
    nd = len(x_in) # get number of data points
    npred = len(xpred_in) # get length of array to use for x predictions
    ypred = np.empty(npred)
    
    # make predictions at the values in xpred_in
    for i in range(0, npred):
        ypred[i] = m_in * xpred_in[i] + b_in
        
    # create a label for the slope and intercept
    # uses string formatting for the numbers
    # the :.2f means floating point number, round to 2 decimal places
    ypred_label = '$m = ' + '{:.2f}'.format(m_in) \
                + '$, $ b = ' +  '{:.2f}'.format(b_in) + '$'
    
    # make plot of predictions with data
    plt.plot(x_in, data_in,'o');
    plt.plot(xpred_in, ypred, '+', color='orange', label=ypred_label)
    plt.xlabel(r'$x$ variable', fontsize=14);
    plt.ylabel(r'$y$ variable', fontsize=14);
    # plot x and y axes to see more easily
    plt.axhline(0,linestyle='--',color='grey');
    plt.axvline(0,linestyle='--',color='grey');
    plt.legend(fontsize=12)

def least_squares_fit_linear(x_in, data_in, m_in, b_in, xpred_in, display_table=True):
    '''
    This function takes in a data set and makes a straight-line prediction.
    It computes the sum of the squared deviations of the predictions from
    the given data set and displays a plot.
    '''
     # get some information from stuff passed in
    nd = len(x_in) # get number of data points
    npred = len(xpred_in) # get length of array to use for x predictions
    ypred = np.empty(npred) # create an array to store model predictions
    
    # make predictions at the values in xpred_in
    for i in range(0, npred):
        ypred[i] = m_in * xpred_in[i] + b_in
        
    # create a label for the slope and intercept to show on the plot
    # uses string formatting for the numbers
    # the :.2f means floating point number, round to 2 decimal places
    ypred_label = '$m = ' + '{:.2f}'.format(m_in) \
                + '$, $ b = ' +  '{:.2f}'.format(b_in) + '$'
    
    # calculate squared deviations and sum them
    squared_devs = np.empty(nd)
    biggest_err = -1
    x_idx_where_biggest = -1
    if display_table:
        print('{: >20}'.format('x value'),'{: >20}'.format('residual squared'))
    for i in range(0, nd):
        abs_err_tmp = abs(data_in[i] - (m_in*x_in[i] + b_in))
        squared_devs[i] = abs_err_tmp**2.0
        if display_table:
            print('{:20.4f}'.format(x_in[i]),'{:20.4f}'.format(squared_devs[i]))
        if abs_err_tmp > biggest_err:
            biggest_err = abs_err_tmp
            x_idx_where_biggest = i
    sum_squared_devs = np.sum(squared_devs)
    print('The sum of squared deviations is', '{:.2f}'.format(sum_squared_devs))
    print('And in case you need to know...')
    print('The largest absolute error is r =', '{:.2f}'.format(biggest_err),'\n',
         'It occurs when x = ', '{:.4f}'.format(x_in[x_idx_where_biggest]))
        
    # make plot of predictions with data
    plt.plot(x_in, data_in,'o');
    plt.plot(xpred_in, ypred, '+', color='orange', label=ypred_label);
    # overlay the plot with the x y pair where the absolute error is biggest
    # plt.plot(x_in[x_idx_where_biggest], data_in[x_idx_where_biggest],'o', color='red');
    plt.xlabel(r'$x$ variable', fontsize=14);
    plt.ylabel(r'$y$ variable', fontsize=14);
    # plot x and y axes to see more easily
    plt.axhline(0,linestyle='--',color='grey');
    plt.axvline(0,linestyle='--',color='grey');
    plt.legend(fontsize=12)

def least_squares_optimizer_fit_linear(x_in, data_in, m_guess, b_guess, xpred_in, display_table=True):
    '''
    This function takes in a data set and makes a straight-line prediction.
    It computes the optimal slope and intercept for the dataset using least-squares optimization.
    It computes the sum of the squared deviations of the predictions from
    the given data set and displays a plot.
    '''
     # get some information from stuff passed in
    nd = len(x_in) # get number of data points
    npred = len(xpred_in) # get length of array to use for x predictions
    ypred = np.empty(npred) # create an array to store model predictions

    # construct the objective function and then minimize it
    def sum_of_lsq(params):
        chi2 = 0.0
        for i in range(0,nd):
            f_x_i = params[0] + params[1] * x_in[i]
            chi2 += (data_in[i]-f_x_i)**2.0
        return chi2
    
    x0 = [b_guess, m_guess]
    res = minimize(sum_of_lsq, x0)

    if res.status == 0:
        print('After', res.nit, 'iterations...')
        m_val = res.x[1]
        b_val = res.x[0]
        print('The optimization was successful! Here are the values it found:')
        print('{: >10}'.format('m ='),'{:10.4f}'.format(res.x[1]))
        print('{: >10}'.format('b ='),'{:10.4f}'.format(res.x[0]))
    else:
        print('The optimization was not successful. Here is the error message:')
        print(res.message)
        print('This problem might be ill-conditioned or you might need to change the starting guesses.')
        return
    
    # make predictions at the values in xpred_in
    for i in range(0, npred):
        ypred[i] = m_val * xpred_in[i] + b_val
        
    # create a label for the slope and intercept to show on the plot
    # uses string formatting for the numbers
    # the :.2f means floating point number, round to 2 decimal places
    ypred_label = '$m = ' + '{:.2f}'.format(m_val) \
                + '$, $ b = ' +  '{:.2f}'.format(b_val) + '$'
    
    # calculate squared deviations and sum them
    squared_devs = np.empty(nd)
    biggest_err = -1
    x_idx_where_biggest = -1
    if display_table:
        print('{: >20}'.format('x value'),'{: >20}'.format('residual squared'))
    for i in range(0, nd):
        abs_err_tmp = abs(data_in[i] - (m_val*x_in[i] + b_val))
        squared_devs[i] = abs_err_tmp**2.0
        if display_table:
            print('{:20.4f}'.format(x_in[i]),'{:20.4f}'.format(squared_devs[i]))
        if abs_err_tmp > biggest_err:
            biggest_err = abs_err_tmp
            x_idx_where_biggest = i
    sum_squared_devs = np.sum(squared_devs)
    print('The sum of squared deviations is', '{:.4f}'.format(sum_squared_devs))
    print('And in case you need to know...')
    print('The largest absolute error is r =', '{:.4f}'.format(biggest_err),'\n',
         'It occurs when x = ', '{:.4f}'.format(x_in[x_idx_where_biggest]))
        
    # make plot of predictions with data
    plt.plot(x_in, data_in,'o');
    plt.plot(xpred_in, ypred, '+', color='orange', label=ypred_label);
    # overlay the plot with the x y pair where the absolute error is biggest
    # plt.plot(x_in[x_idx_where_biggest], data_in[x_idx_where_biggest],'o', color='red');
    plt.xlabel(r'$x$ variable', fontsize=14);
    plt.ylabel(r'$y$ variable', fontsize=14);
    # plot x and y axes to see more easily
    plt.axhline(0,linestyle='--',color='grey');
    plt.axvline(0,linestyle='--',color='grey');
    plt.legend(fontsize=12)

def chebyshev_fit_linear(x_in, data_in, m_in, b_in, xpred_in, display_table=True):
    # get some information from stuff passed in
    nd = len(x_in) # get number of data points
    npred = len(xpred_in) # get length of array to use for x predictions
    ypred = np.empty(npred) # create an array to store model predictions
    
    # make predictions at the values in xpred_in
    for i in range(0, npred):
        ypred[i] = m_in * xpred_in[i] + b_in
        
    # create a label for the slope and intercept to show on the plot
    # uses string formatting for the numbers
    # the :.2f means floating point number, round to 2 decimal places
    ypred_label = '$m = ' + '{:.2f}'.format(m_in) \
                + '$, $ b = ' +  '{:.2f}'.format(b_in) + '$'
    
    # calculate absolute errors and find the biggest one in the list
    abs_err = np.empty(nd)
    biggest_err = -1 # placeholder for biggest error
    x_idx_where_biggest = 0 # placeholder for index of biggest error
    if display_table:
        print('{: >20}'.format('x value'),'{: >20}'.format('residual'))
    for i in range(0, nd):
        abs_err[i] = abs(data_in[i] - (m_in*x_in[i] + b_in))
        if display_table:
            print('{:20.4f}'.format(x_in[i]),'{:20.4f}'.format(abs_err[i]))
        if abs_err[i] > biggest_err:
            biggest_err = abs_err[i]
            x_idx_where_biggest = i
    print('Largest absolute error is r =', '{:.2f}'.format(biggest_err),'\n',
         'It occurs when x = ', '{:.4f}'.format(x_in[x_idx_where_biggest]))
        
    # make plot of predictions with data
    plt.plot(x_in, data_in,'o');
    plt.plot(xpred_in, ypred, '+', color='orange', label=ypred_label);
    # overlay the plot with the x y pair where the absolute error is biggest
    plt.plot(x_in[x_idx_where_biggest], data_in[x_idx_where_biggest],'o', color='red');
    plt.xlabel(r'$x$ variable', fontsize=14);
    plt.ylabel(r'$y$ variable', fontsize=14);
    # plot x and y axes to see more easily
    plt.axhline(0,linestyle='--',color='grey');
    plt.axvline(0,linestyle='--',color='grey');
    plt.legend(fontsize=12)

def chebyshev_linprog_fit_linear(x_in, data_in, xpred_in, m_guess, b_guess):
    nd = len(x_in) # get number of data points
    npred = len(xpred_in)
    ypred = np.empty(npred)
    # construct a matrix to pass into the linprog function
    Aub_mx = np.empty([2*nd, 3]) # matrix with 2*nd rows and 3 columns
    bub_vec = np.empty([2*nd,1]) # a 2*nd element long column vector
    c_vec = np.ones(3) # a list of 3 ones that the linprog function needs

    max_r_val = 1.5*(max(data_in) - min(data_in))
    
    # construct the Aub_mx, use nested for loops
    for i in range(0, 3): # columns
        data_index = 0
        for j in range(0, 2*nd): # rows
            if i == 0: # the first column
                Aub_mx[j,i] = -1
            elif i == 1: # second column (coefficients of m)
                if j%2 == 0: # if j is even
                    Aub_mx[j,i] = -x_in[data_index]
                else: # if j is odd
                    Aub_mx[j,i] = x_in[data_index]
            elif i == 2: # third column (coefficients of b)
                if j%2 == 0:
                    Aub_mx[j,i] = -1
                else:
                    Aub_mx[j,i] = 1
                    
            if j%2 != 0: # increment the data index on odd j values
                data_index += 1
    
    # construct the bub_vec
    data_index = 0
    for j in range(0, 2*nd):
        if j%2 == 0:
            bub_vec[j] = -data_in[data_index]
        else:
            bub_vec[j] = data_in[data_index]
            data_index += 1
    #print(bub_vec)

    m_lo = m_guess-0.33*abs(m_guess)-1e-10
    m_hi = m_guess + 0.33*abs(m_guess)+1e-10
    b_lo = b_guess-0.33*abs(b_guess)-1e-10
    b_hi = b_guess + 0.33*abs(b_guess)+1e-10
    
    # construct bounds for the decision variables
    # constrain m and b close to your guesses to keep the optimizer from wandering off
    bounds = [(0,max_r_val),\
              (m_lo,m_hi),\
              (b_lo, b_hi)]
    
    print('Performing linear program optimization with bounds:')
    print('{: >10}'.format('r ='), '{:10.0f}'.format(0), '{:10.2f}'.format(max_r_val))
    print('{: >10}'.format('m ='), '{:10.2f}'.format(m_lo), '{:10.2f}'.format(m_hi))
    print('{: >10}'.format('b ='), '{:10.2f}'.format(b_lo), '{:10.2f}'.format(b_hi))
    
    res = linprog(c_vec, A_ub=Aub_mx, b_ub=bub_vec, bounds=bounds, method='simplex')
    
    if res.status == 0:
        print('After', res.nit, 'iterations...')
        biggest_err = res.x[0]
        m_val = res.x[1]
        b_val = res.x[2]
        print('The optimization was successful! Here are the values it found:')
        print('{: >10}'.format('r ='),'{:10.4f}'.format(res.x[0]))
        print('{: >10}'.format('m ='),'{:10.4f}'.format(res.x[1]))
        print('{: >10}'.format('b ='),'{:10.4f}'.format(res.x[2]))
    else:
        print('The optimization was not successful. Here is the error message:')
        print(res.message)
        print('This problem might be ill-conditioned or you might need to change the starting guesses.')
        return
    
    ypred_label = '$m = ' + '{:.2f}'.format(m_val) \
                + '$, $ b = ' +  '{:.2f}'.format(b_val) + '$'
    
    # double-check the biggest error calculation
    abs_err = np.empty(nd)
    biggest_err_check = -1
    x_idx_where_biggest = -1
    for i in range(0, nd):
        abs_err[i] = abs(data_in[i] - (m_val*x_in[i] + b_val))
        #print('{:20.4f}'.format(x_in[i]),'{:20.4f}'.format(abs_err[i]))
        if abs_err[i] > biggest_err_check:
            biggest_err_check = abs_err[i]
            x_idx_where_biggest = i
    print('Check: Largest absolute error is r =', '{:.2f}'.format(biggest_err_check),'\n',
         'It occurs when x = ', '{:.4f}'.format(x_in[x_idx_where_biggest]))     
    
    # make predictions at the values in xpred_in
    for i in range(0, npred):
        ypred[i] = m_val * xpred_in[i] + b_val
    
    # make plot of predictions with data
    plt.plot(x_in, data_in,'o');
    plt.plot(xpred_in, ypred, '+', color='orange', label=ypred_label);
    # overlay the plot with the x y pair where the absolute error is biggest
    plt.plot(x_in[x_idx_where_biggest], data_in[x_idx_where_biggest],'o', color='red');
    plt.xlabel(r'$x$ variable', fontsize=14);
    plt.ylabel(r'$y$ variable', fontsize=14);
    # plot x and y axes to see more easily
    plt.axhline(0,linestyle='--',color='grey');
    plt.axvline(0,linestyle='--',color='grey');
    plt.legend(fontsize=12)