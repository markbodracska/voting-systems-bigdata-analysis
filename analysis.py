import pandas as pd
from tqdm import tqdm
from scipy import stats

#observed models
systems = ['plurality', 'borda', 'copeland', 'kemeny_young',
           'schulze', 'minimax', 'irv', 'coombs', 'bucklin', 'ranked_pairs']


def open_system_data(system, cand_num):
    """
    Loads in voting system results from csv file.

    Parameters:
        system(str): name of voting system
        cand_num(int): number of candidates
    Returns:
        df(pd.DataFrame): data with voting system results
    """
    path = 'result-' + str(cand_num) + '/' + system + '_' + str(cand_num) + '.csv'
    df = pd.read_csv(path)
    df.index = df['id']
    df = df.drop('id', axis=1)

    return df


def calc_winner_agreements(systems, cand_num):
    """
    Calculates winner agreements between all pairs of voting systems.
    
    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        pair_winner_agreements(list): winner agreements for all possible pairs of given voting systems
    """
    pair_winner_agreements = []
    for system_1 in tqdm(systems):
        for system_2 in tqdm(systems):
            if system_1 != system_2:
                df_1 = open_system_data(system_1, cand_num)
                df_2 = open_system_data(system_2, cand_num)
                df_1 = df_1.rename(columns = {'1st': '1st_1'}, inplace = False)['1st_1']
                df_2 = df_2.rename(columns = {'1st': '1st_2'}, inplace = False)['1st_2']

                joined_df = pd.concat([df_1, df_2], axis=1, join='inner')

                agree_count = len(joined_df[joined_df['1st_1'] == joined_df['1st_2']])
                agree_perc = agree_count/(len(joined_df))

                pair_winner_agreements.append(((system_1, system_2), agree_perc))

    return pair_winner_agreements


def calc_winner_agreements_on_cond_win(systems, cand_num, cond_win_exists):
    """
    Calculates winner agreements between all pairs of voting systems on elections which contain/do not contain Condorcet-winner.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
        cond_win_exists(bool): check agreements on elections containing/not containing Condorcet-winner
    Returns:
        pair_winner_agreements_on_cond_win(list): winner agreements for all possible pairs of given voting systems on elections which contain/do not contain Condorcet-winner
    """
    cond_win = open_system_data('cond_winner', cand_num)

    pair_winner_agreements_on_cond_win = []
    for system_1 in tqdm(systems):
        for system_2 in tqdm(systems):
            if system_1 != system_2:
                df_1 = open_system_data(system_1, cand_num)
                df_1 = pd.concat([df_1, cond_win], axis=1)
                df_2 = open_system_data(system_2, cand_num)
                df_2 = pd.concat([df_2, cond_win], axis=1)


                df_1 = df_1[df_1['cond_win_exists'] == int(cond_win_exists)]
                df_2 = df_2[df_2['cond_win_exists'] == int(cond_win_exists)]

                df_1 = df_1.rename(columns = {'1st': '1st_1'}, inplace = False)['1st_1']
                df_2 = df_2.rename(columns = {'1st': '1st_2'}, inplace = False)['1st_2']

                joined_df = pd.concat([df_1, df_2], axis=1, join='inner')

                agree_count = len(joined_df[joined_df['1st_1'] == joined_df['1st_2']])
                agree_perc = agree_count/(len(joined_df))

                pair_winner_agreements_on_cond_win.append(((system_1, system_2), agree_perc))

    return pair_winner_agreements_on_cond_win


def calc_cond_win_count(systems, cand_num):
    """
    For all systems, calculates how many times each system chooses the Condorcet-winner.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        cond_count_win_list(list): list of systems with the rate of picking Condorcet-winner
    """
    cond_win = open_system_data('cond_winner', cand_num)
    cond_count_win_list = []

    for system in tqdm(systems):
        df_1 = open_system_data(system, cand_num)
        df_1 = pd.concat([df_1, cond_win], axis=1)

        df_1 = df_1[df_1['cond_win_exists'] == 1]

        cond_count = len(df_1[df_1['1st'] == df_1['cond_win']])
        cond_perc = cond_count/(len(df_1))

        cond_count_win_list.append((system, cond_perc))

    return cond_count_win_list
    

def calc_cond_lose_count(systems, cand_num):
    """
    For all systems, calculates how many times each system chooses the Condorcet-loser.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        cond_count_los_list(list): list of systems with the rate of picking Condorcet-loser
    """
    cond_lose = open_system_data('cond_loser', cand_num)
    cond_count_los_list = []

    for system in tqdm(systems):
        df_1 = open_system_data(system, cand_num)
        df_1 = pd.concat([df_1, cond_lose], axis=1)

        df_1 = df_1[df_1['cond_loser_exists'] == 1]

        cond_count = len(df_1[df_1['1st'] == df_1['cond_loser']])
        cond_perc = cond_count/(len(df_1))

        cond_count_los_list.append((system, cond_perc))

    return cond_count_los_list

def calc_maj_win_count(systems, cand_num):
    """
    For all systems, calculates how many times each system chooses the majority winner.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        maj_count_win_list(list): list of systems with the rate of picking majority winner
    """
    maj_win = open_system_data('maj_winner', cand_num)
    maj_count_win_list = []

    for system in tqdm(systems):
        df_1 = open_system_data(system, cand_num)
        df_1 = pd.concat([df_1, maj_win], axis=1)

        df_1 = df_1[df_1['maj_win_exists'] == 1]

        maj_count = len(df_1[df_1['1st'] == df_1['maj_win']])
        maj_perc = maj_count/(len(df_1))

        maj_count_win_list.append((system, maj_perc))

    return maj_count_win_list
    

def calc_maj_lose_count(systems, cand_num):
    """
    For all systems, calculates how many times each system chooses the majority loser.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        maj_count_los_list(list): list of systems with the rate of picking majority loser
    """
    maj_lose = open_system_data('maj_loser', cand_num)
    maj_count_los_list = []

    for system in tqdm(systems):
        df_1 = open_system_data(system, cand_num)
        df_1 = pd.concat([df_1, maj_lose], axis=1)

        df_1 = df_1[df_1['maj_loser_exists'] == 1]

        maj_count = len(df_1[df_1['1st'] == df_1['maj_loser']])
        maj_perc = maj_count/(len(df_1))

        maj_count_los_list.append((system, maj_perc))

    return maj_count_los_list



def calc_pref_correl(systems, cand_num):
    """
    Calculates correlation between each pair of voting systems according to social preference ordering.

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
    Returns:
        pair_pref_correl(list): rankcorrelation between social preference orderings for all possible pairs of given voting systems
    """
    pair_pref_correl = []

    for system_1 in tqdm(systems):
        for system_2 in tqdm(systems):
            if system_1 != system_2 and systems.index(system_1) < systems.index(system_2):
                df_1 = open_system_data(system_1, cand_num)
                df_2 = open_system_data(system_2, cand_num)
                df_1 = df_1.rename(columns = {'1st': '1st_1', '2nd': '2nd_1', '3rd': '3rd_1', '4th': '4th_1'}, inplace = False)
                df_2 = df_2.rename(columns = {'1st': '1st_2', '2nd': '2nd_2', '3rd': '3rd_2', '4th': '4th_2'}, inplace = False)
                
                joined_df = pd.concat([df_1, df_2], axis=1, join='inner')
                joined_df['corr'] = joined_df.apply(lambda x: stats.spearmanr([x['1st_1'], x['2nd_1'], x['3rd_1'], x['4th_1']], [x['1st_2'], x['2nd_2'], x['3rd_2'], x['4th_2']])[0], axis=1)
                
                corr_mean = joined_df['corr'].mean()
                corr_sd = joined_df['corr'].std()

                pair_pref_correl.append(((system_1, system_2), (corr_mean, corr_sd)))
                
    return pair_pref_correl



def calc_pref_correl_on_cond_win(systems, cand_num, cond_win_exists):
    """
    Calculates rankcorrelation between each pair of voting systems on elections which contain/do not contain Condorcet-winner according to social preference ordering 

    Parameters:
        systems(list): list of voting systems
        cand_num(int): number of candidates
        cond_win_exists(bool): check agreements on elections containing/not containing Condorcet-winner
    Returns:
        pair_pref_correl_on_cond(list): rankcorrelation between social preference orderings for all possible pairs of given voting systems on elections which contain/do not contain Condorcet-winner
    """
    pair_pref_correl_on_cond = []
    cond_win = open_system_data('cond_winner', cand_num)

    for system_1 in tqdm(systems):
        for system_2 in tqdm(systems):
            if system_1 != system_2 and systems.index(system_1) < systems.index(system_2):
                df_1 = open_system_data(system_1, cand_num)
                df_1 = pd.concat([df_1, cond_win], axis=1)
                df_2 = open_system_data(system_2, cand_num)
                df_2 = pd.concat([df_2, cond_win], axis=1)
                
                df_1 = df_1[df_1['cond_win_exists'] == 1]
                df_2 = df_2[df_2['cond_win_exists'] == 1]

                df_1 = df_1.rename(columns = {'1st': '1st_1', '2nd': '2nd_1', '3rd': '3rd_1', '4th': '4th_1'}, inplace = False)
                df_2 = df_2.rename(columns = {'1st': '1st_2', '2nd': '2nd_2', '3rd': '3rd_2', '4th': '4th_2'}, inplace = False)
                
                joined_df = pd.concat([df_1, df_2], axis=1, join='inner')
                joined_df['corr'] = joined_df.apply(lambda x: stats.spearmanr([x['1st_1'], x['2nd_1'], x['3rd_1'], x['4th_1']], [x['1st_2'], x['2nd_2'], x['3rd_2'], x['4th_2']])[0], axis=1)
                
                corr_mean = joined_df['corr'].mean()
                corr_sd = joined_df['corr'].std()

                pair_pref_correl_on_cond.append(((system_1, system_2), (corr_mean, corr_sd)))
                
    return pair_pref_correl_on_cond
