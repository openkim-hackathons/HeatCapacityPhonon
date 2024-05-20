# Import libraries
import os
import re
import json
import numpy as np

# Function for data extraction
def extract_mean_error_from_logfile(filename:str, quantity:int):
    '''
    Function to extract the average from a LAAMPS log file for a given quantity

    @param filename : name of file
    @param quantity : quantity to take from
    @return mean : reported mean value
    '''

    # Get content
    with open(filename, 'r') as file:
        data = file.read()

    # Look for print pattern
    exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
    mean_pattern = r'"mean"\s*([^ ]+)'
    error_pattern = r'"upper_confidence_limit"\s*([^ ]+)'
    match_init = re.search(exterior_pattern, data, re.DOTALL)
    mean_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
    error_matches = re.findall(error_pattern, match_init.group(), re.DOTALL)
    if mean_matches is None:
        raise ValueError('Mean not found')
    if error_matches is None:
        raise ValueError('Error not found')

    # Get correct match
    mean = mean_matches[quantity]
    error = error_matches[quantity]

    # Return
    return mean, error

# Function to compute heat capacity from MD
def compute_heat_capacity(f1:str, f2:str, eps:float, quantity:int):
    '''
    Function to compute heat capacity by finite difference from two simulations
    @param f1 : filename of first file
    @param f2 : filename of second file
    @param eps : epsilon
    @param quantity : number quantity in the log file
    @return c : heat capacity
    @return err : the error in the reported value
    '''

    # Extract mean values
    H_plus_mean, H_plus_err = extract_mean_error_from_logfile(f1)
    H_minus_mean, H_minus_err = extract_mean_error_from_logfile(f2)

    # Compute heat capacity
    c = (H_plus_mean - H_minus_mean) / (2 * eps)

    # Error computation
    dx = H_plus_mean - H_plus_err
    dy = H_minus_mean - H_minus_err
    c_err = np.sqrt(((dx / (2 * eps)) ** 2) - ((dy / (2 * eps)) ** 2))

    # Return
    return c, c_err