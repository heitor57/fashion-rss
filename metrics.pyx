import numpy as np


# @jit(nopython=True
    # )  # Set "nopython" mode for best performance, equivalent to @njit
# def rr(ranks, is_click
      # ):  # Function is compiled to machine code when called the first time
    # for i in range(len(ranks)):  # Numba likes loops
        # if is_click[i] == 1:
            # return ranks[i]

        # # trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    # return 0  # Numba likes NumPy broadcasting


# @jit(nopython=True
    # )  # Set "nopython" mode for best performance, equivalent to @njit
# def mrr(ranks, is_clicks
       # ):  # Function is compiled to machine code when called the first time
    # mrr = 0
    # for i in range(len(ranks)):  # Numba likes loops
        # mrr += rr(ranks[i], is_clicks[i])

        # # trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    # mrr /= len(ranks)
    # return mrr  # Numba likes NumPy broadcasting


def cython_rr(long[:] query_ids,long[:] product_ids,long[:] ranks,dict is_click):

    for i in range(len(ranks)):  # Numba likes loops
        # print((query_ids[i],product_ids[i]),is_click[(query_ids[i],product_ids[i])])
        if is_click[(query_ids[i],product_ids[i])] > 0:
            # print('end')
            # print()('h',1.0/ranks[i])
            return 1.0/ranks[i]
    # print('end here')

    return -99999999
