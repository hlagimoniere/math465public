import numpy as np

def divided_differences(x, y, order):
    m = len(x)
    # for a data set with m elements, there are at most m-1 divided differences 
    # so order < m
    if order >= m:
        print('Error: cannot go to that order of divided differences', order)
        print('Please reduce the order for the data set and try again.')
        return
    
    # compute the divided differences and print them out later
    diffs = []*order # empty list with length of order to hold each diffs array
    for i in range(0, order):
        ndiffs = m-(i+1)
        these_diffs = np.empty(ndiffs)
        if i > 0:
            last_diffs = diffs[i-1]
            for j in range(0,ndiffs):
                these_diffs[j] = (last_diffs[j+1]-last_diffs[j])/(x[j+(i+1)]-x[j]) 
            diffs.append(these_diffs)
        else:
            for j in range(0,ndiffs):
                these_diffs[j] = (y[j+1]-y[j])/(x[j+1]-x[j])
            diffs.append(these_diffs)
    print('{:^21s}'.format('Data'),'|','{:^21s}'.format('Differences'))
    second_line = '{:^21s}'.format('') + ' |'
    for i in range(0, order):
        second_line += '{:^12s}'.format('Delta_' + str(i+1)) + '|'
    print(second_line)
    for i in range(0, m): # iterate over data points for rows
        this_line = '{:^10.2f}'.format(x[i])\
                   + '| ' + '{:^10.2f}'.format(y[i]) +'|'
        if i > 0:
            ncols_fill = min([i,order])
            shift_i = 0
            for j in range(0,ncols_fill):
                #print('diffs[',j,',',i-1-j,'] =', diffs[j][i-1-j])
                this_line += '{:^12.6f}'.format(diffs[j][i-1-j]) + '|'
                shift_i += 1
        print(this_line)