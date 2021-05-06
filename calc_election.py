import numpy as np
from tqdm import tqdm
import statistics

def delete_start(data, cand_num):
    """
    Helper function for read_soc_file(), deletes redundant rows from the data.

    Parameters:
        data(list): processed election data
        cand_num(int): number of candidates
    Returns:
        new_data(list): election data with redundant rows deleted
    """
    new_data = []
    for row in data:
        if (len(row) == cand_num + 1):
            new_data.append(row)

    return new_data


def read_soc_file(file_num, cand_num):
    """
    Reads and processes text file with election data into useable form.

    Parameters:
        file_num(int): id of file
        cand_num(int): number of candidates
    Returns:
        new_data(list): preferences and counts given in lists
    """
    digit_num = len(str(file_num))
    zero_str = '0' * (9-digit_num)

    with open('Netflix-soc-' + str(cand_num) + '/' + zero_str + str(file_num) + "_soc.txt") as f:
        lines = f.read().splitlines()
        data = [line.split(',') for line in lines]

    new_data = delete_start(data, cand_num)

    return new_data



def calc_pref_matrix(data, cand_num):
    """
    Calculates preference matrix from election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        pref_matrix(np.ndarray): preference matrix
    """
    pref_matrix = np.zeros((cand_num,cand_num))

    for pref in data:
        for i in range(1,cand_num):
            for j in range(i+1,len(pref)):
                pref_matrix[int(pref[i])-1][int(pref[j])-1] += int(pref[0])

    return pref_matrix


def calc_margin_matrix(pref_matrix):
    """
    Calculates margin matrix from preference matrix.

    Parameters:
        pref_matrix(np.ndarray): preference matrix
    Returns:
        marg_matrix(np.ndarray): margin matrix
    """
    trans_pref_matrix = np.transpose(pref_matrix)
    marg_matrix = np.subtract(pref_matrix, trans_pref_matrix)

    return marg_matrix