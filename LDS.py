import argparse
import numpy as np
import scipy
from scipy.linalg import pinv, svd  # Your only additional allowed imports!
# python assignment3.py -f dt1_train.npy -q 5 -o outfile.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3",
                                     epilog="CSCI 4360/6360 Data Science II: Fall 2017",
                                     add_help="How to use",
                                     prog="python assignment3.py <arguments>")
    parser.add_argument("-f", "--infile", required=True,
                        help="Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required=True, type=int,
                        help="Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # Read in the dynamic texture data.
    M = np.load(input_file)
    print('M',M.shape)
    # f x h x w -> f x hw
    Y = np.array([m.flatten() for m in M]).transpose()
    
    # Singular value decompositon for dimentionality reduction
    U, S, V = np.linalg.svd(Y, full_matrices=False)
    print('U',U.shape,'S',S.shape,'V',V.shape)

    # get q number of frames
    S = np.diag(S[:q])
    C = U[:,:q]
    X = np.dot(S,V[:q])

    # calculate pseudo-inverse
    A = np.linalg.pinv(X)

    A = A[-1:]
    X = X[:,-1:]

    # simulate next time step
    X_new = np.dot(A,X)

    # State space to Appearance space
    Y_new = np.dot(C,X)
    # hw x f -> f x h x w
    M_new = Y_new.reshape(M.shape[1:])
    print(M_new.shape)
    # (110,160)
    np.save(output_file, M_new)
    print('A',A.shape)

    print('Y', Y.shape)
    print('X', X.shape)
    print('C', C.shape)

    print('X_new',X_new.shape)
    print('Y_new',Y_new.shape)
    print('M_new',M_new.shape)



