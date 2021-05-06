import numpy as np
import pandas as pd
import calc_election
from tqdm import tqdm

#conversion between numerical and string candidate representation
cand_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}


def calc_first_place_votes(data, cand_num):
    """
    Calculates first place votes for each candidate in the election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        first_place_votes(np.ndarray): array of first place votes
    """
    first_place_votes = np.zeros((cand_num,))
    
    for pref in data:
        first_place_votes[int(pref[1])-1] += int(pref[0])

    return first_place_votes


def plurality(data, cand_num):
    """
    Calculates plurality voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    first_place_votes = calc_first_place_votes(data, cand_num)
    winner = first_place_votes.argmax() 

    return winner


def calc_borda_scores(data, cand_num):
    """
    Calculates Borda-scores for each candidate in the election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        borda_scores(np.ndarray): array of Borda-scores
    """
    borda_scores = np.zeros((cand_num,))

    for pref in data:
        for i in range(1, len(pref)):
            borda_scores[int(pref[i])-1] += (len(pref)-i)*int(pref[0])

    return borda_scores

def borda(data, cand_num):
    """
    Calculates Borda voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    borda_scores = calc_borda_scores(data, cand_num)
    winner = borda_scores.argmax()

    return winner


def calc_win_matrix(margin_matrix):
    """
    Calculates win matrix from given margin matrix.

    Parameters:
        margin_matrix(np.ndarray): margin matrix
    Returns:
        win_matrix(np.ndarray): win matrix
    """
    win_matrix = np.sign(margin_matrix)

    return win_matrix


def check_deleted_cands(data, cand_num):
    """
    Checks which candidates have been deleted from the election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        deleted_cands(list): deleted candidates
    """
    if cand_num == 4:
        all_cands = ['1', '2', '3', '4']
    elif cand_num == 5:
        all_cands = ['1', '2', '3', '4', '5']

    remaining_cands = data[0][1:]

    deleted_cands = (set(all_cands)).difference(set(remaining_cands))
    deleted_cands = list(deleted_cands)

    return deleted_cands

def copeland(data, cand_num):
    """
    Calculates Copeland voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    deleted_cands = check_deleted_cands(data, cand_num)

    pref_matrix = calc_election.calc_pref_matrix(data, cand_num)
    margin_matrix = calc_election.calc_margin_matrix(pref_matrix)
    win_matrix = calc_win_matrix(margin_matrix)

    for element in deleted_cands:
        win_matrix[int(element)-1] = np.nan

    winner = np.nanargmax(win_matrix.sum(axis=1))

    return winner


def check_condorcet_winner(margin_matrix, cand_num):
    """
    Checks if there exists a Condorcet-winner in given election.

    Parameters:
        margin_matrix(np.ndarray): margin matrix of election
        cand_num(int): number of candidates
    Returns:
        cond_win_exists(bool): gives whether there is a Condorcet-winner in the election
        cond_win(string): if exists, Condorcet-winner, if not, gives nan
    """
    cond_win_exists = False
    cond_win = np.nan
    win_matrix = calc_win_matrix(margin_matrix)
    sums = win_matrix.sum(axis=1)

    for element in sums:
        if (element == cand_num + 1):
            cond_win_exists = True
            cond_win = cand_dict[int(np.where(sums == element)[0]) + 1]

    return cond_win_exists, cond_win


def permutation(lst):
    """
    Calculates all permutations of given list.

    Parameters:
        lst(list): list of candidates
    Returns:
        l(list): list of permutations
    """
    if len(lst) == 0:
        return []

    if len(lst) == 1:
        return [lst]
  
    l = [] 

    for i in range(len(lst)):
       m = lst[i]
       remLst = lst[:i] + lst[i+1:]
       for p in permutation(remLst):
           l.append([m] + p)
           
    return l
  
def kemeny_young(data, cand_num):
    """
    Calculates Kemeny-Young voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    remaining_cands = data[0][1:]

    pref_list = permutation(remaining_cands)
    pref_matrix = calc_election.calc_pref_matrix(data, cand_num)
    margin_matrix = calc_election.calc_margin_matrix(pref_matrix)

    final_pref_list = []
    for pref in pref_list:
        value = 0
        for el in pref[:-1]:
            for el_after in pref[(pref.index(el)+1):]:
                value += margin_matrix[int(el)-1][int(el_after)-1]
        
        pref.append(value)
        final_pref_list.append(pref)

        max_value = 0
        winner = ''

        for pref in final_pref_list:
            if (pref[-1] > max_value):
                max_value = pref[-1]
                winner = pref[0]

        if (winner == ''):
            winner = remaining_cands[-1]
    
    return int(winner) - 1


def calc_p(pref_matrix, candidates):
    """
    Calculates array of strongest paths for given election.

    Parameters:
        pref_matrix(np.ndarray): preference matrix
        candidates(list): list of candidates
    Returns:
        stro_paths(dict): dictionary of strongest paths
    """
    stro_paths = {}
    for candidate_name1 in candidates:
        for candidate_name2 in candidates:
            if candidate_name1 != candidate_name2:
                strength = pref_matrix[int(candidate_name1)-1][int(candidate_name2)-1]
                if strength > pref_matrix[int(candidate_name2)-1][int(candidate_name1)-1]:
                    stro_paths[candidate_name1, candidate_name2] = strength

    
    for candidate_name1 in candidates:
        for candidate_name2 in candidates:
            if candidate_name1 != candidate_name2:
                for candidate_name3 in candidates:
                    if (candidate_name1 != candidate_name3) and (candidate_name2 != candidate_name3):
                        curr_value = stro_paths.get((candidate_name2, candidate_name3), 0)
                        new_value = min(
                                stro_paths.get((candidate_name2, candidate_name1), 0),
                                stro_paths.get((candidate_name1, candidate_name3), 0))
                        if new_value > curr_value:
                            stro_paths[candidate_name2, candidate_name3] = new_value

    return stro_paths

def rank_p(candidates, stro_paths):
    """
    Calculates preference ordering from array of strongest paths.

    Parameters:
        candidates(list): list of candidates
        stro_paths(dict): strongest paths
    Returns:
        winner(int): numerical id of winner candidate
    """
    candidate_wins = []

    for candidate_name1 in candidates:
        num_wins = 0

        for candidate_name2 in candidates:
            if candidate_name1 == candidate_name2:
                continue
            candidate1_score = stro_paths.get((candidate_name1, candidate_name2), 0)
            candidate2_score = stro_paths.get((candidate_name2, candidate_name1), 0)
            if candidate1_score > candidate2_score:
                num_wins += 1

        candidate_wins.append((candidate_name1, num_wins))

    candidate_wins.sort(reverse = True, key=lambda tup: tup[1])
    winner = [cand_win[0] for cand_win in candidate_wins][0]

    return winner


def schulze(data, cand_num):
    """
    Calculates Schulze voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    remaining_cands = data[0][1:]
    pref_matrix = calc_election.calc_pref_matrix(data, cand_num)

    p = calc_p(pref_matrix, remaining_cands)
    winner = rank_p(remaining_cands, p)

    return int(winner)-1


def minimax(data, cand_num):
    """
    Calculates Minimax voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    deleted_cands = check_deleted_cands(data, cand_num)

    pref_matrix = calc_election.calc_pref_matrix(data, cand_num)
    margin_matrix = calc_election.calc_margin_matrix(pref_matrix)

    worst_defeats = margin_matrix.min(axis = 1)

    for element in deleted_cands:
        worst_defeats[int(element)-1] = np.nan

    winner = np.nanargmax(worst_defeats)

    return winner



def check_maj_winner_by_first_places(first_places):
    """
    Checks if there exists a majority winner in given election.

    Parameters:
        first_places(np.ndarray): array of first place votes
    Returns:
        maj_winner_exists(bool): gives whether there is a majority winner in the election
    """
    maj_winner_exists = np.nanmax(first_places) >= (np.nansum(first_places)/2)
    return maj_winner_exists


def delete_cand(data, candidate):
    """
    Deletes candidate from election data.

    Parameters:
        data(list): election data
    Returns:
        updated_data(list): election data with given candidate deleted
    """
    data = [[ele for ele in sub if ele != str(candidate) or sub.index(ele) == 0] for sub in data]
    
    updated_data = []
    for pref in data:
        if (len([i for i in range(len(pref)) if pref[i] in candidate]) > 1):
            new_pref = list(dict.fromkeys(pref))
            updated_data.append(new_pref)
        else:
            updated_data.append(pref)
    
    return updated_data


def irv(data, cand_num):
    """
    Calculates IRV voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    ori_deleted_cands = check_deleted_cands(data, cand_num)

    deleted_cands = []

    winner_found = False
    while not winner_found:
        first_place_votes = calc_first_place_votes(data, cand_num)
        for cand in ori_deleted_cands:
            first_place_votes[int(cand)-1] = np.nan
        if check_maj_winner_by_first_places(first_place_votes):
            winner = np.nanargmax(first_place_votes)
            winner_found = True
        else:
            for el in first_place_votes:
                int_deleted_cands = [int(i) for i in deleted_cands]
                array_cands = (np.where(first_place_votes == el)[0]+1)
                if (el == 0) and (set(array_cands)).difference(set(int_deleted_cands)) == set():
                    first_place_votes[np.where(first_place_votes == el)[0]] = np.nan
                    
            worst_candidate = np.nanargmin(first_place_votes) + 1
            deleted_cands.append(worst_candidate)
            data = delete_cand(data, str(worst_candidate))

    return winner



def calc_last_place_votes(data, cand_num):
    """
    Calculates last place votes for each candidate in the election data.

    Parameters:
        data(list): election 
        cand_num(int): number of candidates
    Returns:
        last_place_votes(np.ndarray): array of last place votes
    """
    last_place_votes = np.zeros((cand_num,))
    
    for pref in data:
        last_place_votes[int(pref[-1])-1] += int(pref[0])

    return last_place_votes


def coombs(data, cand_num):
    """
    Calculates Coombs voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    deleted_cands = []

    winner_found = False
    while not winner_found:
        first_place_votes = calc_first_place_votes(data, cand_num)
        last_place_votes = calc_last_place_votes(data, cand_num)
        if check_maj_winner_by_first_places(first_place_votes):
            winner = first_place_votes.argmax()
            winner_found = True
        else:       
            worst_candidate = np.nanargmax(last_place_votes) + 1
            deleted_cands.append(worst_candidate)
            data = delete_cand(data, str(worst_candidate))

    return winner


def calc_bucklin_points(data, order, cand_num):
    """
    Calculates Bucklin-scores on given level of order for each candidate.

    Parameters:
        data(list): election data
        order(int): level of order
        cand_num(int): number of candidates
    Returns:
        bucklin_points(np.ndarray): array of Bucklin-scores
    """
    bucklin_points = np.zeros((cand_num,))
    for i in range(1,order+1):
        for pref in data:
            bucklin_points[int(pref[i])-1] += int(pref[0])

    return bucklin_points


def check_winner_by_majority(bucklin_points, order):
    """
    Checks if there exists a majority winner on given level of order in given election.

    Parameters:
        bucklin_points(np.ndarray): array of Bucklin-scores
        order(int): level of order
    Returns:
        maj_winner_exists(bool): gives whether there is a majority winner on the level of order
    """
    vote_count = bucklin_points.sum()/order
    maj_winner_exists = bucklin_points.max() >= vote_count/2

    return maj_winner_exists

def bucklin(data, cand_num):
    """
    Calculates Bucklin voting winner on election data.

    Parameters:
        data(list): election data
    Returns:
        winner(int): numerical id of winner candidate
    """
    winner_found = False
    order = 1
    while not winner_found:
        bucklin_points = calc_bucklin_points(data, order, cand_num)
        if check_winner_by_majority(bucklin_points, order):
            winner = bucklin_points.argmax()
            winner_found = True
        else:
            order += 1

    return winner


def get_pair_strength(pair, margin_matrix):
    """
    Calculates strength of given pair of candidates.

    Parameters:
        pair(tuple): two candidates
        margin_matrix(np.ndarray): margin matrix of election
    Returns:
        pair_strength(int): margin strength of given pair
    """
    pair_strength = margin_matrix[int(pair[0])-1][int(pair[1])-1]
    return pair_strength


def calc_pair_graph(candidates, pairs):
    """
    Creates pair graph from given pairs.

    Parameters:
        candidates(list): future nodes of pair graph
        pairs(list): future edges of pair graph
    Returns:
        winners(set): set of nodes where no node has a parent -> winner
    """
    edges = set()
    children = set()

    for (i, j) in pairs:
        if i in candidates and j in candidates and i not in children and (j, i) not in edges:
            children.add(j)
            edges.add((i, j))

    winners = set()
    for c in candidates:
        if c not in children:
            winners.add(c)

    
    return winners



def ranked_pairs(data, cand_num):
    """
    Calculates ranked pairs voting winner on election data.

    Parameters:
        data(list): election data
        cand_num(int): number of candidates
    Returns:
        winner(int): numerical id of winner candidate
    """
    remaining_cands = data[0][1:]
    pref_matrix = calc_election.calc_pref_matrix(data, cand_num)
    margin_matrix = calc_election.calc_margin_matrix(pref_matrix)

    pairs = [(i, j) for i in remaining_cands for j in remaining_cands if i != j]
    pair_strengths = [(pair, get_pair_strength(pair, margin_matrix)) for pair in pairs if get_pair_strength(pair, margin_matrix) >= 0]
    pair_strengths.sort(reverse = True, key=lambda tup: tup[1])

    final_pair_list = [pair[0] for pair in pair_strengths]
    winners = calc_pair_graph(remaining_cands, final_pair_list)

    winner = int(list(winners)[0]) - 1

    return winner


def check_condorcet_loser(margin_matrix, cand_num):
    """
    Checks if there exists a Condorcet-loser in given election.

    Parameters:
        margin_matrix(np.ndarray): margin matrix of election
        cand_num(int): number of candidates
    Returns:
        cond_loser_exists(bool): gives whether there is a Condorcet-loser in the election
        cond_lose(string): if exists, Condorcet-loser, if not, gives nan
    """
    cond_loser_exists = False
    cond_lose = np.nan
    win_matrix = calc_win_matrix(margin_matrix)
    sums = win_matrix.sum(axis=1)

    for element in sums:
        if (element == cand_num*(-1)+1):
            cond_loser_exists = True
            cond_lose = cand_dict[int(np.where(sums == element)[0]) + 1]

    return cond_loser_exists, cond_lose


def get_social_pref_ord(file_num, voting_sys, cand_num):
    """
    Creates social preference ordering for given election on given voting system.

    Parameters:
        file_num(int): id of file
        voting_sys(str): name of voting system
        cand_num(int): number of candidates
    Returns:
        soc_pref_order_list(list): social preference ordering
    """
    data = calc_election.read_soc_file(file_num, cand_num)

    soc_pref_order_list = []
    process_done = False
    while not process_done:
        if len(data[0]) > 2:
            winner = voting_sys(data)
            soc_pref_order_list.append(cand_dict[winner+1])
            data = delete_cand(data, str(winner+1))
        else:
            soc_pref_order_list.append(cand_dict[int(data[0][1])])
            process_done = True

    return soc_pref_order_list
