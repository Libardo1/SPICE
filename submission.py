# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:47:00 2016

@author: Remi Eyraud

Tested with Python 2.7 and 3.5
"""

# State the problem number
problem_number = '0'

# and the user id (given during registration)
user_id = ''

train_file = '0.spice.train'
prefix_file = '0.spice.public.test'

# name of this submission (only letters, numbers and _)
name = "My_algo_"

def learn(train_file, parameter):
    """ Put here the learning part """
    return list()


def next_symbols_ranking(model, prefix):
    """ Put here the ranking computation """
    return "3 2 1 0 -1"


def get_first_prefix(test_file):
    """ get the only prefix in test_file """
    f = open(test_file)
    prefix = f.readline()
    f.close()
    return prefix


def formatString(string_in):
    """ Replace white spaces by %20 """
    return string_in.strip().replace(" ", "%20")

# learn the model
print ("Start Learning")
model = learn(train_file, 0)
print ("End of learning phase")

# get the test first prefix: the only element of the test set
first_prefix = get_first_prefix(prefix_file)

# get the next symbol ranking on the first prefix
ranking = next_symbols_ranking(model, first_prefix)

print("Prefix number: 1 Prefix: " + first_prefix + " Ranking: " + ranking)

# transform ranking to follow submission format (with %20 between symbols)
ranking = formatString(ranking)

# transform the first prefix to follow submission format
first_prefix = formatString(first_prefix)

# create the url to submit the ranking
url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id +\
           '&problem=' + problem_number + '&submission=' + name + '&'
url = url_base + 'prefix=' + first_prefix + '&prefix_number=1' + '&ranking=' +\
      ranking

# Get the website answer for the first prefix with this ranking using this
# submission name
try:
    # Python 2.7
    import urllib2 as ur
    orl2 = True
except:
    #Python 3.4
    import urllib.request as ur
    orl2 = False

response = ur.urlopen(url)
content = response.read()
if not orl2:
    # Needed for python 3.4...
    content= content.decode('utf-8')

list_element = content.split()
head = str(list_element[0])

prefix_number = 2

while(head != '[Error]' and head != '[Success]'):
    # Get rid of Line feed
    prefix = content[:-1]
    
    # Get the ranking
    ranking = next_symbols_ranking(model, prefix)
    
    print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking + " Prefix: " + prefix)
    
    # Format the ranking
    ranking = formatString(ranking)

    # create prefix with submission needed format
    prefix=formatString(prefix)

    # Create the url with your ranking to get the next prefix
    url = url_base + 'prefix=' + prefix + '&prefix_number=' +\
        str(prefix_number) + '&ranking=' + ranking

    # Get the answer of the submission on current prefix
    response = ur.urlopen(url)
    content = response.read()
    if not orl2:
        # Needed for python 3.4...
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
