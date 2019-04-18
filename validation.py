# Jeśli chcemy przetestować wybrany katalog, należy podać jego ścieżkę jako argument
# Domyślnie sprawdzamy cały folder ./data/

import glob
import os
import sys
import main

root = "./data/**/"
model_filename = "correct.txt"

correct = 0
wrong = 0

root = "{}/**/".format(sys.argv[1]) if len(sys.argv) > 1 else root

correct_files = glob.glob(os.path.join(root, model_filename), recursive=True)

for correct_file in correct_files:
    set_dir = os.path.dirname(correct_file)
    n_images = len([image for image in os.listdir(set_dir)]) - 1
    try:
        with open(correct_file) as model:
            results = main.run_alg(set_dir, n_images)
            for image in results:
                model_line = model.readline()
                if model_line.rstrip() == str(image[0]):
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
