# -*- coding: utf-8 -*-
"""
    Created in Thu March  22 10:47:00 2016

    @author: Remi Eyraud & Sicco Verwer

    Tested with Python 2.7.11 and Python 3.4
"""

# State the problem number
problem_number = '0'

# and the user id (given during registration)
user_id = ''

# name of this submission (no space or special character)

name = "n_gram"
NULL = -1
train_file = '1.spice.train.txt'
prefix_file = '1.spice.public.test.txt'
ngramModel = 5

from decimal import *

import numpy as np


def number(arg):
    return Decimal(arg)


def ngramCounts(sequences, N):
    counts = {}
    for n in range(1, N + 1):
        nCounts = {}
        seq = [tuple([NULL] * (n - 1) + sequence) for sequence in sequences]
        ngrams = [y for x in [zip(*[sequence[i:] for i in range(n)]) for sequence in seq] for y in x]
        for ngram in ngrams:
            if ngram[:-1] in nCounts:
                if ngram[-1] in nCounts[ngram[:-1]]:
                    nCounts[ngram[:-1]][ngram[-1]] += 1
                else:
                    nCounts[ngram[:-1]][ngram[-1]] = 1
            else:
                nCounts[ngram[:-1]] = {}
                nCounts[ngram[:-1]][ngram[-1]] = 1
        counts[n] = nCounts

    counts[0] = {}
    counts[0][()] = {}
    counts[0][()][()] = sum(counts[1][()].values())
    return counts


#  with smoothing
def logProb(Wn, Wp, counts, lambdas):
    assert len(Wn) == 1 and len(Wp) + 1 == len(lambdas)
    W = tuple(Wn) + tuple(Wp)
    wCounts = []
    for k in range(len(W), -1, -1):
        if k == 0:
            wCounts += [counts[0][()][()]]
        else:
            wCounts += [counts[len(W[0:k])][W[0:k - 1]][W[k - 1]]]
    wProbs = []
    for k in range(len(wCounts) - 1):
        if wCounts[k] == 0:
            wProbs += [0]
        else:
            wProbs += [float(wCounts[k]) / wCounts[k + 1]]
    return np.log(np.sum(np.array(lambdas) * np.array(wProbs)))


def ngramrank(prefix, alphabet, counts, lambdas, N):
    Wp = prefix[-(N - 1):]
    rank = {}
    for alpha in alphabet:
        rank[alpha] = logProb([alpha], Wp, counts, lambdas)

    return sorted(rank, key=rank.get, reverse=True)


def readset(f):
    sett = []
    line = f.readline()
    l = line.split(" ")
    num_strings = int(l[0])
    alphabet_size = int(l[1])
    for n in range(num_strings):
        line = f.readline()
        l = line.split(" ")
        sett = sett + [[int(i) for i in l[1:len(l)]]]
    return alphabet_size, sett


def get_first_prefix(test_file):
    """ get the only prefix in test_file """
    f = open(test_file)
    prefix = f.readline()
    f.close()
    return prefix


def list_to_string(l):
    s = str(l[0])
    for x in l[1:]:
        s += " " + str(x)
    return (s)


def formatString(string_in):
    """ Replace white spaces by %20 """
    return string_in.strip().replace(" ", "%20")


print("Get training sample")
alphabet, train = readset(open('../data/train/' + train_file, "r"))
print ("Start Learning")
lambdas = [0.5, 0.3, 0.1, 0.08, 0.02]
counts = ngramCounts(train, ngramModel)

print ("Learning Ended")

# get the test first prefix: the only element of the test set
first_prefix = get_first_prefix('../data/test_public/' + prefix_file)
prefix_number = 1

# get the next symbol ranking on the first prefix
p = first_prefix.split()
prefix = [int(i) for i in p[1:len(p)]]
ranking = ngramrank(prefix, range(alphabet), counts, lambdas, ngramModel)
ranking_string = list_to_string(ranking[:5])

print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + first_prefix)

# transform the first prefix to follow submission format
first_prefix = formatString(first_prefix)

# transform the ranking to follow submission format
ranking_string = formatString(ranking_string)

# create the url to submit the ranking
url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id + \
           '&problem=' + problem_number + '&submission=' + name + '&'
url = url_base + 'prefix=' + first_prefix + '&prefix_number=1' + '&ranking=' + \
      ranking_string

# Get the website answer for the first prefix with this ranking using this
# submission name
try:
    # Python 2.7
    import urllib2 as ur

    orl2 = True
except:
    # Python 3.4
    import urllib.request as ur

    orl2 = False

response = ur.urlopen(url)
content = response.read()

if not orl2:
    # Needed for python 3.4...
    content = content.decode('utf-8')

list_element = content.split()
head = str(list_element[0])

prefix_number = 2

while (head != '[Error]' and head != '[Success]'):
    prefix = content[:-1]
    # Get the ranking
    p = prefix.split()
    prefix_list = list()
    prefix_list = [int(i) for i in p[1:len(p)]]
    ranking = threegramrank(prefix_list, alphabet, dict)
    ranking_string = list_to_string(ranking[:5])

    print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + prefix)

    # Format the ranking
    ranking_string = formatString(ranking_string)

    # create prefix with submission needed format
    prefix = formatString(prefix)

    # Create the url with your ranking to get the next prefix
    url = url_base + 'prefix=' + prefix + '&prefix_number=' + \
          str(prefix_number) + '&ranking=' + ranking_string

    # Get the answer of the submission on current prefix
    response = ur.urlopen(url)
    content = response.read()
    if not orl2:
        # Needed for Python 3.4...
        content = content.decode('utf-8')

    list_element = content.split()
    # modify head in case it is finished or an erro occured
    head = str(list_element[0])
    # change prefix number
    prefix_number += 1

# Post-treatment
# The score is the last element of content (in case of a public test set)
print(content)

list_element = content.split()
score = (list_element[-1])
print(score)
print 'hello'
