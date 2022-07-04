import os
import pandas as pd
import Levenshtein


def leven_dist_to__(cs0):
    def dist(cs1):
        return Levenshtein.distance(
            cs0, cs1)
    return dist

def closest_word_leven_dist_to__(cs0):
    sg_dist = leven_dist_to__(cs0)
    def dist(words):
        return min(
            map(sg_dist, words))
    return dist

def string_data__(original, words):
    return {
        'og': original,
        'words': words}

def augment_dict(data, key, value):
    data[key] = value
    return data

def strings_data__(strings):
    return list(
        map(lambda tup: string_data__(*tup),
            zip(strings,
                map(lambda cs: cs.split(' '),
                    strings))))

def augment_levenshtein_distance_to_strings_data(
        strings_data, pattern):
    pass
    # dist_f = leven_dist_to__(pattern)
    # def aug(dat):
    #     return augment_dict(
    #         dat, 'dist', dat['og']
    # return list(
    #     map(
    

if __name__ == '__main__':
    print('string_similarity.py\n')

    df_articole = pd.read_csv(
        "/".join([
            'transfer_date_mircea',
            'intrari_articole_ardeleanu__20_06_2022.csv']))

    denumiri = list(set(
        map(lambda cs: cs.lower(),
            df_articole['denumire'].dropna())))

    # wordss = list(
    #     map(lambda cs: cs.split(' '),
    #         denumiri))

    strings_data = strings_data__(denumiri)

    paine = 'paine'

    dist = closest_word_leven_dist_to__(paine)
    
    strings_data.sort(
        key=lambda dat: dist(dat['words']))

