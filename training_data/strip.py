import json
import os
import gzip
from urllib.request import urlopen

data = []
with gzip.open(urlopen('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Amazon_Review.json.gz')) as f:
    for l in f:
        data.append(json.loads(l.strip()))

f_out = open(f"training_data_{filename.rstrip('.json')}.txt", "w+")
for i in data:
    try:
        big_giant_text_blob = d["reviewText"].rstrip("\n") + " "
        f_out.write(big_giant_text_blob)
    except KeyError:
        continue
f_out.close()

# This one extracts from the jsons # forget this bit
"""
directory = os.fsencode(os.getcwd())
# https://stackoverflow.com/a/10378012/9295513
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"):
        f = open(filename, "r")
        l = f.readlines()
        f_out = open(f"training_data_{filename.rstrip('.json')}.txt", "w+")
        for i in l:
            d = json.loads(i)
            # print(i)
            try:
                big_giant_text_blob = d["reviewText"].rstrip("\n") + " "
                f_out.write(big_giant_text_blob)
            except KeyError:
                continue
        f_out.close()
        f.close()
    print(f"{filename} is done.")
# """
# The following two utilities are plagiarized verbatim from their cited sources
# This is a machine learning course, not a "how to clean data" course
# The first is there because Books.json was too big to load into memory
# The second because the above outputs multiple text files, one per .json.

# This one was meant particularly for Books
"""
# https://stackoverflow.com/a/16290885/9295513
lines_per_file = 500
smallfile = None
with open('Books.json') as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'training_data_Books_{}.json'.format(lineno + lines_per_file)
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()
"""

# This one is meant to combine all the text files.
"""
# https://stackoverflow.com/a/27077437/9295513
import shutil

with open('all_training_data.txt','wb') as wfd:
    for f in os.listdir(os.fsencode(os.getcwd())):
        if f.endswith(b".txt"):
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
# """