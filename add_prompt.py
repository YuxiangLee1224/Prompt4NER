# Hard prompts

import json
from collections import Counter


def get_entity():
    with open('prompt_data/MIT_MM/new_10_shot.json','r') as fr:
        json_data = json.load(fr)

        json_data = json_data['atis']
        for i in range(10):
            data = json_data[i]['support']['prompt_seq_out']
            for j in range(len(data)):
                ind = data[j][0].index("'", 2) + 4

                for x in range(19):
                    # res = data[j][x][ind:]
                    if x == 11 or x == 3:
                        res = data[j][x][ind+1:]
                    else: res = data[j][x][ind:]
                    if ' '.join(res) != 'none':
                        mrc_entity[x].append(' '.join(res))

def get_top_5_elements(lst):
    for i in range(19):
        counts = Counter(lst[i])
        top_5 = counts.most_common(5)
        res[i] = [item[0] for item in top_5]


def pre_add_prompt():
    res = {"atis":[{},{},{},{},{},{},{},{},{},{}]}
    x = 0
    with open('prompt_data/MIt_MM/org_10_shot.json', 'r') as fp:
        json_data = json.load(fp)
        # len(json_data['atis']) ----> 10
        for data in json_data['atis']:

            res["atis"][x]["domain"] = "atis"
            res["atis"][x]["support"] = {}
            res["atis"][x]["test"] = {}

            query_data = data['test']
            qosi = query_data['original_seq_in']
            qoso = query_data['original_seq_out']
            qobso = query_data['original_bio_seq_out']
            qpsi = query_data['prompt_seq_in']
            qpso = query_data['prompt_seq_out']


            support_data = data['support']
            osi = support_data['original_seq_in']
            oso = support_data['original_seq_out']
            obso = support_data['original_bio_seq_out']
            psi = support_data['prompt_seq_in']
            pso = support_data['prompt_seq_out']


            for i in range(len(psi)):
                for j in range(19):
                    psi[i][j] = mrc_query[j] + psi[i][j]
                    pso[i][j] = mrc_query[j] + pso[i][j]

            res["atis"][x]["support"]['original_seq_in'] = osi
            res["atis"][x]["support"]['original_seq_out'] = oso
            res["atis"][x]["support"]['original_bio_seq_out'] = obso
            res["atis"][x]["support"]['prompt_seq_in'] = psi
            res["atis"][x]["support"]['prompt_seq_out'] = pso

            for i in range(len(qosi)):
                for j in range(19):
                    qpsi[i][j] = mrc_query[j] + qpsi[i][j]
                    qpso[i][j] = mrc_query[j] + qpso[i][j]

            res["atis"][x]["test"]['original_seq_in'] = qosi
            res["atis"][x]["test"]['original_seq_out'] = qoso
            res["atis"][x]["test"]['original_bio_seq_out'] = qobso
            res["atis"][x]["test"]['prompt_seq_in'] = qpsi
            res["atis"][x]["test"]['prompt_seq_out'] = qpso

            x = x + 1

    with open("prompt_data/MIT_M/new_10_shot.json", 'w') as write_f:
        json.dump(res, write_f)

def post_add_prompt():
    res = {"atis":[{},{},{},{},{},{},{},{},{},{}]}
    x = 0
    with open('prompt_data/MIT_M/org_20_shot.json', 'r') as fp:
        json_data = json.load(fp)
        # len(json_data['atis']) ----> 10
        for data in json_data['atis']:

            res["atis"][x]["domain"] = "atis"
            res["atis"][x]["support"] = {}
            res["atis"][x]["test"] = {}

            query_data = data['test']
            qosi = query_data['original_seq_in']
            qoso = query_data['original_seq_out']
            qobso = query_data['original_bio_seq_out']
            qpsi = query_data['prompt_seq_in']
            qpso = query_data['prompt_seq_out']

            support_data = data['support']
            osi = support_data['original_seq_in']
            oso = support_data['original_seq_out']
            obso = support_data['original_bio_seq_out']
            psi = support_data['prompt_seq_in']
            pso = support_data['prompt_seq_out']

            for i in range(len(psi)):
                length = len(osi[i]) + 2
                for j in range(12):
                    psi[i][j] = psi[i][j][:length] + mrc_query[j] +psi[i][j][length:]
                    pso[i][j] = pso[i][j][:length] + mrc_query[j] +pso[i][j][length:]

            res["atis"][x]["support"]['original_seq_in'] = osi
            res["atis"][x]["support"]['original_seq_out'] = oso
            res["atis"][x]["support"]['original_bio_seq_out'] = obso
            res["atis"][x]["support"]['prompt_seq_in'] = psi
            res["atis"][x]["support"]['prompt_seq_out'] = pso

            for i in range(len(qosi)):
                length = len(qosi[i]) + 2
                for j in range(12):
                    qpsi[i][j] = qpsi[i][j][:length] + mrc_query[j] + qpsi[i][j][length:]
                    qpso[i][j] = qpso[i][j][:length] + mrc_query[j] + qpso[i][j][length:]

            res["atis"][x]["test"]['original_seq_in'] = qosi
            res["atis"][x]["test"]['original_seq_out'] = qoso
            res["atis"][x]["test"]['original_bio_seq_out'] = qobso
            res["atis"][x]["test"]['prompt_seq_in'] = qpsi
            res["atis"][x]["test"]['prompt_seq_out'] = qpso

            x = x + 1

    with open("prompt_data/MIT_M/new_20_shot.json", 'w') as write_f:
        json.dump(res, write_f)

if __name__ == "__main__":
    mrc_entity = [[], [], [], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], []]
    res = [[], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [],[], [], []]
    mrc_query = [
        ['actor', 'is', 'an', 'entity', 'such', 'as'],
        ['award', 'is', 'an', 'entity', 'such', 'as'],
        ['character', 'is', 'an', 'entity', 'such', 'as'],
        ['character', 'name', 'is', 'an', 'entity', 'such', 'as'],
        ['director', 'is', 'an', 'entity', 'such', 'as'],
        ['genre', 'is', 'an', 'entity', 'such', 'as'],
        ['opinion', 'is', 'an', 'entity', 'such', 'as'],
        ['origin', 'is', 'an', 'entity', 'such', 'as'],
        ['plot', 'is', 'an', 'entity', 'such', 'as'],
        ['quote', 'is', 'an', 'entity', 'such', 'as'],
        ['rating', 'is', 'an', 'entity', 'such', 'as'],
        ['ratings', 'average', 'is', 'an', 'entity', 'such', 'as'],
        ['relationship', 'is', 'an', 'entity', 'such', 'as'],
        ['reviews', 'is', 'an', 'entity', 'such', 'as'],
        ['song', 'is', 'an', 'entity', 'such', 'as'],
        ['soundtrack', 'is', 'an', 'entity', 'such', 'as'],
        ['title', 'is', 'an', 'entity', 'such', 'as'],
        ['trailer', 'is', 'an', 'entity', 'such', 'as'],
        ['year', 'is', 'an', 'entity', 'such', 'as']
    ]
    strList = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    for i, temp in enumerate(res):
        for x, data in enumerate(temp):
            str = data.split(' ')
            strList[i] += str
            strList[i].append(',')
        res[i] = strList[i]

    for i, temp in enumerate(res):
        mrc_query[i] += res[i]
        mrc_query[i].append('.')
        print(mrc_query[i])

    # get_entity()
    # get_top_5_elements(mrc_entity)
    # for i in range(len(res)):
    #     print(res[i])
    pre_add_prompt()





