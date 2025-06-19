import pandas as pd
from random import sample, randint, choice
import numpy as np
import string
import multiprocessing as mp
from tqdm import tqdm
import unidecode
import datetime
import argparse
import configparser
import os

def parse_generator_configuration():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cf", "--configuration_file", type=str, help='specify path to configuration file')
    args = parser.parse_args()

    config = configparser.ConfigParser()

    config.read(args.configuration_file)

    return config

def split_horizontal(source: pd.DataFrame, common: float, pick_unique: bool):
    columns = source.columns.tolist()
    _, col_num = source.shape
    common = max(common, 1)

    if pick_unique:
        columns_rows = [(col, len(set(source[col].dropna()))) for col in columns] # sort columns by number of unique values 
        columns_rows.sort(key=lambda tup: tup[1], reverse=True)

        common_columns = [tup[0] for tup in columns_rows[:common]]
    else:
        common_columns = sample(columns, common)

    col_num -= common
    store = source[common_columns].copy()

    temp = source.drop(common_columns, axis=1)

    temp_cols = temp.columns.tolist()

        
    remaining_cols = temp_cols
    temp = temp[remaining_cols]

    random_slice = int(len(remaining_cols)/2)

    generated_1 = store.join(temp.iloc[:, :random_slice], lsuffix='pk')
    generated_2 = store.join(temp.iloc[:, random_slice:], lsuffix='pk')

    return generated_1, generated_2, common_columns

def split_vertical(source_1: pd.DataFrame, source_2: pd.DataFrame, overlap: float, typos: bool, cov_prc:int, typo_prc:int, cascade:bool):
    records_num, _ = source_2.shape

    overlapping_rows = int(overlap*records_num)

    random_slice = randint(1, records_num-overlapping_rows-1)

    target_1 = source_1.iloc[:overlapping_rows + random_slice]
    target_2 = pd.concat([source_2.iloc[:overlapping_rows], source_2.iloc[overlapping_rows + random_slice:]])

    tmp = source_2.iloc[:overlapping_rows]
    if typos:
        cols_to_update = list(set(target_1.columns.tolist()).intersection(set(target_2.columns.tolist())))
        if cascade:
            target_2 = update_values(target_2, cov_prc, typo_prc, cols_to_update)
        else:
            tmp = update_values(tmp, cov_prc, typo_prc, cols_to_update)
            target_2 = pd.concat([tmp, source_2.iloc[overlapping_rows + random_slice:]])

    return target_1, target_2

letters = list(string.ascii_lowercase)

proximity = {'a': ['q', 'w', 'z', 's', 'x'],
             'b': ['f', 'g', 'h', 'v', 'n'],
             'c': ['s', 'x', 'd', 'f', 'v'],
             'd': ['w', 'e', 'r', 's', 'x', 'c', 'f', 'v'],
             'e': ['w', 'r', 's', 'd', 'f', '2', '3', '4'],
             'f': ['e', 'r', 't', 'd', 'c', 'b', 'g', 'v'],
             'g': ['r', 't', 'y', 'f', 'b', 'h', 'v', 'n'],
             'h': ['t', 'y', 'u', 'b', 'm', 'j', 'g', 'n'],
             'i': ['u', 'o', 'j', 'k', 'l', '7', '8', '9'],
             'j': ['y', 'u', 'i', 'm', 'h', 'k', 'n'],
             'k': ['u', 'i', 'o', 'm', 'j', 'l'],
             'l': ['i', 'o', 'p', 'k'],
             'm': ['j', 'h', 'k', 'n'],
             'n': ['b', 'm', 'j', 'g', 'h'],
             'o': ['i', 'p', 'k', 'l', '8', '9', '0'],
             'p': ['o', 'l', '9', '0'],
             'q': ['w', 'a', 's', '1', '2'],
             'r': ['e', 't', 'd', 'f', 'g', '3', '4', '5'],
             's': ['q', 'w', 'e', 'a', 'z', 'x', 'd', 'c'],
             't': ['r', 'y', 'f', 'g', 'h', '4', '5', '6'],
             'u': ['y', 'i', 'j', 'h', 'k', '6', '7', '8'],
             'v': ['d', 'c', 'f', 'b', 'g'],
             'w': ['q', 'e', 'a', 's', 'd', '1', '2', '3'],
             'x': ['a', 'z', 's', 'd', 'c'],
             'y': ['t', 'u', 'j', 'g', 'h', '5', '6', '7'],
             'z': ['a', 's', 'x'],
             '1': ['q', 'w', '2'],
             '2': ['q', 'w', 'e', '1', '3'],
             '3': ['w', 'e', 'r', '2', '4'],
             '4': ['e', 'r', 't', '3', '5'],
             '5': ['r', 't', 'y', '4', '6'],
             '6': ['t', 'y', 'u', '5', '7'],
             '7': ['y', 'u', 'i', '6', '8'],
             '8': ['u', 'i', 'o', '7', '9'],
             '9': ['i', 'o', 'p', '8', '0'],
             '0': ['o', 'p', '9']
             }


def clean_word(word: str):
    alphanumerics = []
    non_alpha = dict()
    for i, val in enumerate(word):
        if val.isalnum():
            alphanumerics.append(val)
        else:
            non_alpha[i] = val

    result = "".join(alphanumerics)
    return result, non_alpha


def proximity_typo(letter: str):
    return choice(proximity[letter])


def perturb_data(data: np.ndarray):
    tmp = data[~np.isnan(data)]
    sign = sample([-1, 1], 1)[0]
    perturb = sign * randint(10, 50) / 100

    mu = np.mean(tmp)
    st_dev = np.std(tmp)

    mu = mu + mu * perturb
    st_dev = st_dev + st_dev * perturb

    tmp2 = np.random.normal(mu, st_dev, tmp.size)
    tmp2 = np.absolute(tmp2)

    if (tmp % 1 == 0).all():
        tmp2 = np.around(tmp2)
    to_return = data.copy()
    count = 0
    for i in range(len(to_return)):
        if pd.isnull(to_return[[i]]):
            continue
        else:
            to_return[i] = tmp2[count]
            count += 1

    return to_return


def sub_job(args: tuple):
    col = args[0]
    frame = args[1]
    approx_prc = args[2]
    typeCol = args[3]
    rec_to_update = args[4]
    cols_to_update = args[5]


    if not cols_to_update or col in cols_to_update:
        regex = datetime.datetime.strptime

        example = frame[0]
        choice = 0
        try:
            assert regex(example, '%m/%d/%Y')
            choice = 1
        except:
            try:
                assert regex(example, '%m/%d/%Y %I:%M:%S %p')
                choice = 2
            except:
                try:
                    assert regex(example, '%A, %B %d, %Y')
                    choice = 3
                except:
                    try:
                        assert regex(example, '%I:%M %p')
                        choice = 4
                    except:
                        try:
                            assert regex(example, '%Y-%m-%dT%H:%M:%S.000Z')
                            choice = 5
                        except:
                            choice = 0

        if choice: # perturb dates/times
            random_indices = sample(range(len(frame)), rec_to_update)
            for rec in random_indices:
                if choice == 1:
                    try:
                        a = regex(frame[rec], '%m/%d/%Y')

                        frame[rec] = a.date().strftime("%A %d %B %Y")
                    except:
                        continue

                elif choice == 2:
                    try:
                        a = regex(frame[rec], '%m/%d/%Y %I:%M:%S %p')
                        frame[rec] = a.date().strftime("%A %d %B %Y")

                    except:
                        continue
                elif choice == 3:
                    try:
                        a = regex(frame[rec], '%A, %B %d, %Y')
                        frame[rec] = a.date().strftime("%m/%d/%Y")
                    except:
                        continue
                elif choice == 4:
                    try:
                        a = regex(frame[rec], '%I:%M %p')
                        frame[rec] = a.time().isoformat()[:5]
                    except:
                        continue
                else:
                    try:
                        a = regex(frame[rec], '%Y-%m-%dT%H:%M:%S.000Z')
                        frame[rec] = '{0} {1}'.format(a.date().strftime('%m/%d/%Y'), a.time().strftime('%I:%M %p'))
                    except:
                        continue

            return col, frame
        elif typeCol == np.dtype('float64'):
            np_frame = np.array(frame)
            new_frame = perturb_data(np_frame)
            return col, new_frame
        elif not typeCol == 'object':
            return col, frame
        else:
            random_indices = sample(range(len(frame)), rec_to_update)
            for rec in random_indices:

                frame[rec] = unidecode.unidecode(str(frame[rec]))

                if pd.isnull([frame[rec]]):
                    continue
                elif frame[rec].replace('-', '').isnumeric():
                    frame[rec] = frame[rec].replace('-', ' ')
                else:
                    try:
                        cleaned_word, nonalphanumerics = clean_word(frame[rec])
                        word = list(cleaned_word)
                    except Exception as e:
                        print(frame[rec])
                        print(col, word)
                        raise Exception(e)
                    hml = max(int(len(word) * approx_prc / 100), 1)
                    try:
                        ltc = sample(range(len(word)), hml)
                    except:
                        ltc = []
                    for l in ltc:
                        pick = proximity_typo(word[l].lower())
                        while word[l] == pick:
                            pick = sample(letters, 1)[0]

                        word[l] = pick
                    word = "".join(word)

                    for index, symbol in nonalphanumerics.items():
                        word = word[:index] + symbol + word[index:]
                    frame[rec] = word
    return col, frame


def update_values(frame: pd.DataFrame, cov_prc: int, typo_prc: int, cols_to_update=None):
    records_num, _ = frame.shape

    rec_to_update = int(records_num * cov_prc / 100)

    pool = mp.Pool(processes=mp.cpu_count() - 1)
    output = dict(pool.imap_unordered(sub_job, input_generator(frame, typo_prc, rec_to_update, cols_to_update)))
    srt = {b: i for i, b in enumerate(list(frame))}
    output = dict(sorted(output.items(), key=lambda t: srt[t[0]]))
    pool.close()
    pert_pd = pd.DataFrame.from_dict(output)
    return pert_pd


def input_generator(frame: pd.DataFrame, typo_prc: int, rec_to_update, cols_to_update):
    for col in frame.columns.tolist():
        yield col, frame[col].tolist(), typo_prc, str(frame[col].dtype), rec_to_update, cols_to_update


if __name__ == '__main__':
    config = parse_generator_configuration()

    test_tables_path = config['PATHS']['test_tables_path']
    train_tables_path = config['PATHS']['train_tables_path']

    num_iterations = config['PARAMETERS'].getint('num_iterations')

    common_cols_l = config['PARAMETERS'].getint('common_cols_l')
    common_cols_r = config['PARAMETERS'].getint('common_cols_r')
    common_cols_step = config['PARAMETERS'].getint('common_cols_step')

    overlaps_l = config['PARAMETERS'].getfloat('overlaps_l')
    overlaps_r = config['PARAMETERS'].getfloat('overlaps_r')
    overlaps_step = config['PARAMETERS'].getfloat('overlaps_step')

    typo_prc = config['PARAMETERS'].getint('typo_prc')
    cov_prc = config['PARAMETERS'].getint('cov_prc')

    common_cols = list(range(common_cols_l, common_cols_r, common_cols_step))
    overlaps = []
    while overlaps_l <= overlaps_r:
        overlaps.append(overlaps_l)
        overlaps_l += overlaps_step

    for i in range(num_iterations):
        for filename in os.listdir(test_tables_path):
            filepath = os.path.join(test_tables_path, filename)
            
            test = pd.read_csv(filepath)
            
            cc = min(choice(common_cols), len(test.columns))
            
            train_1_mid, train_2_mid, _ = split_horizontal(test, cc, pick_unique=choice([True, False]))
            
            overlap = choice(overlaps)
            
            train_1, train_2 = split_vertical(train_1_mid, train_2_mid, overlap=overlap, typos=choice([True, False]), cov_prc=cov_prc, typo_prc=typo_prc, cascade=choice([True, False]))
            
            train_1.to_csv(os.path.join(train_tables_path, f'{filename[:-4]}_{2*i}.csv'), index=False)
            train_2.to_csv(os.path.join(train_tables_path, f'{filename[:-4]}_{2*i+1}.csv'), index=False)

