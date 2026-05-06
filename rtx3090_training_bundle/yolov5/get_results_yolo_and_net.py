from predict_digits import predict_digit, class_mapper as digit_mapper
from predict_letters import predict_letter, class_mapper as letter_mapper
from find_boxes import get_prediction_at_row, get_boxes, get_cropped_image
import json

# calles find_boxes and generates labels (digit & letter) for each box
# requires additional models (predict_digits, predict_letters)
# returns a list of (digit, letter)
# get_result_list(img_file) called from java

img_file = '/Users/wiggel/Python/eKlausurData/Temp/186652_page_5.jpg'

agnostic=True
def get_letters_above_digit(digit, list_of_letters):
    letters_above = []
    width = digit["x2"] - digit["x1"]
    cx = digit["x1"] + width / 2
    for letter in list_of_letters:
        lw = letter["x2"] - letter["x1"]
        lx = letter["x1"] + lw / 2
        if abs(lx - cx) < width * 2:
            # print("--- --->", digit, letter)
            letters_above.append(letter)
    return letters_above


def get_best_letter(list_of_letters):
    list_of_letters2 = []
    for item in list_of_letters:
        if item["letter"] == "?": item["prop"] = 0.1
        list_of_letters2.append(item)
    if list_of_letters2 == []:
        list_of_letters2 = [{'letter': '?', 'prop': 0.0}]

    res = max(list_of_letters2, key=lambda x: x["prop"])

    return res

def generate_lists_of_elements(img_file):
    results = get_boxes(img_file, agnostic)
    list_of_letter = []
    list_of_digits = []
    idx = 0
    for obj in results.pred[0]:
        ycat, x1, y1, x2, y2, conf = get_prediction_at_row(results, idx, False)
        img = get_cropped_image(img_file, x1, y1, x2, y2)
        letter, lprop = predict_letter(img, False)
        digit, dprop = predict_digit(img, False)
        # digit in (123456789)
        if 9 > digit >= 0 and dprop > 0.5:
            # print("d--->", digit_mapper(digit), dprop)
            digit = {"digit": digit_mapper(digit),
                     "prop": dprop.item(),
                     "x1": x1,
                     "y1": y1,
                     "x2": x2,
                     "y2": y2}
            list_of_digits.append(digit)
        else:
            # print("q--->", letter_mapper(letter), lprop)
            letter = {"letter": letter_mapper(letter),
                      "prop": lprop.item(),
                      "x1": x1,
                      "y1": y1,
                      "x2": x2,
                      "y2": y2}
            list_of_letter.append(letter)
        idx += 1
    return list_of_letter, list_of_digits


def get_filtered_and_sorted_list_of_digits(list_of_digits, x1, y1, x2, y2):
    # digits only inside area
    # best digits only, but we use agnostic == True
    # if agnotic == False, we obtain an error if different digits are found in one box
    # sorted list
    list_of_digits2 = (digit for digit in list_of_digits if digit['x1'] >= x1 and
                       digit['x2'] <= x2 and digit['y1'] >= y1 and digit['y2'] <= y2)
    sorted_list_of_digits = sorted(list_of_digits2, key=lambda x: x['digit'])
    '''
    # not working correctly -> switch agnostic to True
    filted_list_of_digits = []
    if sorted_list_of_digits:
        for item in sorted_list_of_digits:
            best_digit = max(sorted_list_of_digits, key=lambda x: x["prop"] and x["digit"] == item['digit'])
            if best_digit not in filted_list_of_digits:
                filted_list_of_digits.append(best_digit)
    '''
    return sorted_list_of_digits


def get_result_list(file, x1=0, y1=0, x2=100000, y2=100000):
    # x1 ... y2 are used to remove digits out of table
    list_of_letters, list_of_digits = generate_lists_of_elements(file)
    sorted_list_of_digits = get_filtered_and_sorted_list_of_digits(list_of_digits, x1, y1, x2, y2)
    result = []
    for digit in sorted_list_of_digits:
        letters = get_letters_above_digit(digit, list_of_letters)
        max_letter = get_best_letter(letters)
        result.append({"digit": digit, "letter": max_letter})
    res_json = json.dumps(result)
    return res_json


def test(str):
    return 'ok: ' + str


def main():
    res = get_result_list(img_file)
    print(res)


if __name__ == "__main__":
    main()
