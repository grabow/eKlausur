import os
import yaml
import numpy as np
from botocore.config import Config
import shutil

#Kopiert alle page-files mit content in das targetverzeichnis
from_path = '/Users/wiggel/NotenCalc'


my_config = Config(region_name='eu-central-1')
#%%
def fetch_sol_from_page(from_dir, stud_folder, page):
    pagefile = "page_" + str(page) + ".jpg"
    filepath = os.path.join(from_dir , stud_folder , pagefile)
    print("get sol for page " , filepath)
    return 'c,d,f'
#%%
def gen_corr_sol(from_path, stud_folder):
    filepath = os.path.join(from_path, stud_folder, 'studSolution.yaml')
    with open(filepath, 'r') as stream:
        studSol = yaml.safe_load(stream)
    sol_str = ''
    sl_list = studSol['solutions']
    idx = 0
    for sol in sl_list:
        sol_str += sl_list[idx] + ","
        idx +=1
    print("Solution-Str:" , sol_str)

def gen_comp_sol(from_path, stud_folder):
    with open(from_path + "/imageMapping.yaml", 'r') as stream:
        imageMapping = yaml.safe_load(stream)
    im_list = imageMapping['mappingList'].items()
    comp_str = ""
    for k, pages in im_list:
        print(k, "->", pages)
        for page in pages:
         comp_str += fetch_sol_from_page(from_path, stud_folder, page) + ","
    print("Solution-Str2:" , comp_str)
#%%
# The following Python function implements the Levenshtein distance in a recursive way
# https://python-course.eu/applications-python/levenshtein-distance.php#:~:text=Enrol%20here-,The%20Minimum%20Edit%20Distance%20or%20Levenshtein%20Dinstance,of%20insertions%2C%20deletions%20and%20substitutions.
def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t

        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """

    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost) # substitution

    # for r in range(rows):
        # print(dist[r])


    return dist[row][col]
#%%
print( iterative_levenshtein("abcd", "cdab") )
#%%
def list_files(dir):
    folders = []
    for r,d,f in os.walk(dir):
        for name in d:
            folders.append(name)
    return folders


folderList = list_files(from_path)
for folder in folderList[0:1]:
    print(folder)
    gen_comp_sol(from_path, folder)
    gen_corr_sol(from_path, folder)