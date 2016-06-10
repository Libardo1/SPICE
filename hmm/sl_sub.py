# -*- coding: utf-8 -*-
"""
    Created in Thu March  22 10:47:00 2016
    
    @author: Remi Eyraud & Sicco Verwer
    
    Tested with Python 2.7.11 and Python 3.4
"""

# State the problem number
problem_number = '3'

# and the user id (given during registration)
user_id = '103'

# name of this submission (no space or special character)

name = "spectral_learning_27"


train_path = '../data/train/'
test_public_path = '../data/test_public/'

train_file = train_path + problem_number + ".spice.train.txt"
prefix_file = test_public_path +  problem_number + ".spice.public.test.txt"

from numpy import *
from decimal import *
from sys import *
from SL_HMM import SLHMM  

def number(arg):
    return Decimal(arg)


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
    s=str(l[0])
    for x in l[1:]:
        s += " " + str(x)
    return(s)

def formatString(string_in):
    """ Replace white spaces by %20 """
    return string_in.strip().replace(" ", "%20")

def slranks(sym_cnt, ):
    sl = SLHMM()

print("Get training sample")
alphabet, train = readset(open(train_file,"r"))
print ("Start Learning")

nStates = 5
slhmm_model = SLHMM(nStates,alphabet+1)

slhmm_model.learn(train)

print ("Learning Ended")

# get the test first prefix: the only element of the test set
first_prefix = get_first_prefix(prefix_file)
prefix_number=1

# get the next symbol ranking on the first prefix
p=first_prefix.split()
prefix=[int(i) for i in p[1:len(p)]]
ranking = slhmm_model.get_rankings(prefix)
ranking_string=list_to_string(ranking[:5])

print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + first_prefix)

# transform the first prefix to follow submission format
first_prefix = formatString(first_prefix)

# transform the ranking to follow submission format
ranking_string=formatString(ranking_string)

# create the url to submit the ranking
url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id +\
    '&problem=' + problem_number + '&submission=' + name + '&'
url = url_base + 'prefix=' + first_prefix + '&prefix_number=1' + '&ranking=' +\
    ranking_string

# Get the website answer for the first prefix with this ranking using this
# submission name    # Python 2.7
import urllib2 as ur
orl2 = True


response = ur.urlopen(url)
content = response.read()

if not orl2:
    # Needed for python 3.4...
    content= content.decode('utf-8')

list_element = content.split()
head = str(list_element[0])

prefix_number = 2

while(head != '[Error]' and head != '[Success]'):
    prefix = content[:-1]
    # Get the ranking
    p=prefix.split()
    prefix_list= list()
    prefix_list=[int(i) for i in p[1:len(p)]]
    ranking = slhmm_model.get_rankings(prefix_list)
    ranking_string=list_to_string(ranking[:5])
    
    print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + prefix)
    
    # Format the ranking
    ranking_string = formatString(ranking_string)
    
    # create prefix with submission needed format
    prefix=formatString(prefix)
    
    # Create the url with your ranking to get the next prefix
    url = url_base + 'prefix=' + prefix + '&prefix_number=' +\
        str(prefix_number) + '&ranking=' + ranking_string
    
    # Get the answer of the submission on current prefix
    response = ur.urlopen(url)
    content = response.read()
    if not orl2:
        # Needed for Python 3.4...
        content= content.decode('utf-8')
    
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

