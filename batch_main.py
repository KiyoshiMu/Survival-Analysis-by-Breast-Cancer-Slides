import os
import numpy as np

if __name__ == "__main__":
    # os.system('python main.py c:/selected -o c:/yes51205 -a 10 -r 0.5 -d 512 -m c:/yes51205/9.h5 -t 31')
    # os.system('python main.py c:/selected -o "e:/no51205" -a 0 -r 0.5 -d 512')
    # os.system('python main.py c:/selected -o "e:/no25608" -a 0 -r 0.8 -m e:/JunkCreator/no25608/105.h5 -t 295')
    # os.system('python main.py c:/selected -o "e:/no25608" -a 0 -r 0.8 -m e:/no25608/600.h5 -t 200')
    # os.system('python main.py c:/selected -m e:/no25608/598.h5 -v 1 -p 1')
    # os.system('python main.py c:/selected -o e:/no51208+ -m e:/no51208+/396.h5 -a 0 -d 512 -r 0.8 -t 3')
    for r in np.arange(0.6, 0.8, 0.1):
        r = round(r , 1)
        os.system(f'python main.py c:/selected -o e:/range/no256{r} -m e:/range/no256{r}/4.h5 -a 0 -d 256 -r {r} -t 35')