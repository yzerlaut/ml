import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


if __name__=='__main__':

    import sys, os

    # visualization module
    sys.path.append('../..')
    from graphs.my_graph import graphs
    mg = graphs('screen')

