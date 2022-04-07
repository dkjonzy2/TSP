import math
import random
import signal
import sys
import time

from Proj5GUI import Proj5GUI
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    w = Proj5GUI()
    times = []
    iterations = 5
    size = 15

    for i in range(iterations):
        w.randSeedClicked()
        w.size = QLineEdit(str(size))
        w.generateClicked()
        w.solveClicked()
        times.append(w.solvedIn.text())
        print(w.tourCost.text(), w.curSeed.text())
    print("Times for running B&B %d times with size %d" % (iterations, size))
    print(times)
    w.close()
    sys.exit(app.exec())