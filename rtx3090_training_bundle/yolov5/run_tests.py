import json
import os
from collections import Counter

import cv2
import numpy as np
import yaml
import PIL.Image

import sys
sys.path.append('/Users/wiggel/Python/llm/llm')  # Add the folder containing quickstart.py to sys.path
from openai_quickstart import copy_blurr_resize  # Now import the function
from openai_quickstart import recognize


# Switching between all and net:
# 1. Change import from get_results_yolo_and_net <-> from get_results_yolo_all
# 2. Adjust in fetch_sol_from_page copy_invert_blurr <-> copy_blurr
# 3. In find_boxes switch the model


from get_results_yolo_all import get_result_list

# from get_results_yolo_and_net import get_result_list

# Adapt for old version of exams
# New Version of imagemappimng.yaml (page-1):  comp_str += fetch_sol_from_page(from_path, stud_folder, page-1, x1, y1, x2, y2)

'''
base_dir = '/workspace/SSD/eKlausurData'
img_file = base_dir + '/Test/179568_page_10_inv_grey.jpg'
test_dir = base_dir + '/Test'
noten_dir = base_dir + '/NotenCalc'
temp_dir = base_dir + '/Temp'
'''

base_dir = '/Users/wiggel'
# img_file = base_dir + '/Python/eKlausurData/Temp/179568_page_3.jpg'
# test_dir = base_dir + '/Python/eKlausurData/Test'
noten_dir = base_dir + '/NotenCalc'
temp_dir = base_dir + '/Python/eKlausurData/Temp'


def list_files(dir):
    files = []
    for r, d, f in os.walk(dir):
        for name in f:
            files.append(name)
    return files


def list_folder(dir):
    folders = []
    for r, d, f in os.walk(dir):
        for name in d:
            folders.append(name)
    return folders


def test_get_list_of_digits(res):
    res_data = json.loads(res)
    dig_list = []
    for dig in res_data:
        nr = dig["digit"]["digit"]
        dig_list = dig_list + [int(nr)]
    print(dig_list)
    return dig_list


def test_get_list_of_letters(res):
    res_data = json.loads(res)
    letter_list = ""
    for dig in res_data:
        abc = dig["letter"]
        if abc == {}: continue
        letter = dig["letter"]["letter"]
        letter_list = letter_list + letter + ","
    return letter_list


def test_check_double_digit(digit_list):
    if digit_list == []:
        return True
    dig_histo = Counter(digit_list).values()
    mm = max(dig_histo)
    return (mm == 1)


def check_test_dir(test_dir):
    list = sorted(list_files(test_dir))
    for file in list:
        if file == '.DS_Store': continue  # F*** Mac
        if file == 'classes.txt': continue  # End
        filepath = os.path.join(test_dir, file)

        res = get_result_list(filepath)
        digits = test_get_list_of_digits(res)
        letters = test_get_list_of_letters(res)
        # print(digits)
        # print(letters)
        if not test_check_double_digit(digits):
            print("--- ---> Digit Error: ", file)


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

    rows = len(s) + 1
    cols = len(t) + 1
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
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)  # substitution

    # for r in range(rows):
    # print(dist[r])

    return dist[row][col]


def gen_corr_sol(from_path, stud_folder):
    filepath = os.path.join(from_path, stud_folder, 'studSolution.yaml')
    with open(filepath, 'r') as stream:
        studSol = yaml.safe_load(stream)
    sol_str = ''
    sl_list = studSol['solutions']
    idx = 0
    for sol in sl_list:
        sol_str += sl_list[idx] + ","
        idx += 1
    sol_str = sol_str.replace(' ', '')
    print("Solution-Str:", sol_str)
    return sol_str


def smooth_filter(img, size):
    img = np.array(img)
    out_img = cv2.GaussianBlur(img, (size, size), 0)
    return out_img


def smooth_filter2(img, size):
    img = np.array(img)
    out_img = cv2.medianBlur(img, size)
    return out_img


def copy_invert_blurr(from_dir, stud_folder, pagefile):
    src = os.path.join(from_dir, stud_folder, pagefile)
    dst = os.path.join(temp_dir, stud_folder + "_" + pagefile)
    print(src + "-->" + dst)
    img = PIL.Image.open(src).convert('L')
    img2 = np.array(img)
    colors_num = np.unique(img2).shape[0]
    if colors_num < 256:
        img3 = 255 - smooth_filter(img2, 9)
    else:
        img3 = 255 - smooth_filter(img2, 7)
    img4 = img3
    img5 = PIL.Image.fromarray(img4)
    img5.save(dst, quality=100, subsampling=0)
    return dst


def copy_blurr_bfw(from_dir, stud_folder, pagefile):
    src = os.path.join(from_dir, stud_folder, pagefile)
    dst = os.path.join(temp_dir, stud_folder + "_" + pagefile)
    # print(src + "-->" + dst)
    img = PIL.Image.open(src).convert('L')
    img2 = np.array(img)
    colors_num = np.unique(img2).shape[0]
    if colors_num < 256:
        img3 = smooth_filter(img2, 3)
    else:
        img3 = smooth_filter(img2, 3)
        img3 = img2
    img4 = img3
    img5 = PIL.Image.fromarray(img4)
    img5.save(dst, quality=100, subsampling=0)
    return dst

def fetch_sol_from_page(from_dir, stud_folder, page, x1=0, y1=0, x2=10000, y2=10000):
    pagefile = "page_" + str(page) + ".jpg"

    # yolo_all:
    # filepath = copy_invert_blurr(from_dir, stud_folder, pagefile)
    # yolo_and_net:
    # filepath = copy_blurr_bw(from_dir, stud_folder, pagefile)

    # for yoloAll and yolo+net use get_result_list( ... )
    # res = get_result_list(filepath, x1, y1, x2, y2)

    # for llm use
    filepath = copy_blurr_resize(from_dir, stud_folder, pagefile)
    res = recognize( filepath )

    # print("get sol for page " , filepath)
    digits = test_get_list_of_digits(res)
    if not test_check_double_digit(digits):
        print("--- ---> Digit Error: ", filepath)
    print(res)
    letters = test_get_list_of_letters(res)
    return letters



def gen_comp_sol(from_path, stud_folder, x1=0, y1=0, x2=10000, y2=10000):
    with open(from_path + "/imageMapping.yaml", 'r') as stream:
        imageMapping = yaml.safe_load(stream)
    im_list = imageMapping['mappingList'].items()
    comp_str = ""
    for k, pages in im_list:
        # print(k, "->", pages)
        for page in pages:
            # New Version of imageemappimng:  comp_str += fetch_sol_from_page(from_path, stud_folder, page-1, x1, y1, x2, y2)
            comp_str += fetch_sol_from_page(from_path, stud_folder, page - 1, x1, y1, x2, y2)
    # print("Solution-Str2:" , comp_str)
    return comp_str


def get_conda_env():
    return os.environ.get("CONDA_DEFAULT_ENV")

def main():


    # fetch_sol_from_page('/Users/wiggel/Python/eKlausurData/Temp','000111',7, 134 , 106, 1294, 230)

    '''
    s = fetch_sol_from_page('/Users/wiggel/NotenCalc','Temp',0)
    print(s)


    # check_test_dir(test_dir)
    # Adjust fetch_sol_from_page (copy_invert_blurr - all, copy_blurr - net)

    '''
    error_sum = 0
    all_sum = 0
    digit_sum = 0
    folderList = sorted(list_folder(noten_dir))
    folderList = ['182114']
    for folder in folderList:
        print(folder)
        sol1 = gen_corr_sol(noten_dir, folder).upper()
        # Specify aerea to get digits
        # sol2 = gen_comp_sol(noten_dir, folder, 134 , 106, 1294, 477).upper()
        sol2 = gen_comp_sol(noten_dir, folder).upper()
        error = iterative_levenshtein(sol1, sol2)
        print("correct:  ", sol1)
        print("computed: ", sol2)
        print(error)
        digit_sum += len(sol2.replace(",", ""))
        all_sum += len(sol1.replace(",", ""))
        error_sum += error
    print("Errors: ", error_sum, " / ", all_sum, ",  Digits recognized: ", digit_sum)


if __name__ == "__main__":
    main()
