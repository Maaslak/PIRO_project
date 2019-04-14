# Jeśli chcemy przetestować wybrany katalog, należy podać jego ścieżkę jako argument
# Domyślnie sprawdzamy cały folder ./data/

import glob
import os
import sys

root = "./data/**/"
model_filename = "correct.txt"
result_filename = "result.txt"

correct = 0
wrong = 0

root = "{}/**/".format(sys.argv[1]) if len(sys.argv) > 1 else root

model_files = glob.glob(os.path.join(root, model_filename), recursive=True)

for model_name in model_files:
    try:
        with open(model_name) as model:
            result_name = os.path.join(os.path.dirname(model_name), result_filename)
            with open(result_name) as result:
                for model_line in model:
                    result_line = result.readline()
                    result_image = result_line.split()[0]
                    if model_line.rstrip() == result_image:
                        correct += 1
                    else:
                        wrong += 1
    except Exception as e:
        print(e)

try:
    print("{0} / {1} images are correct - {2:0.2f}%. "
        .format(correct, correct + wrong, (float(correct) / (correct + wrong)) * 100))
except Exception as e:
    print(e)
