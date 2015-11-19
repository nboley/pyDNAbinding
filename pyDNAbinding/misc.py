def logistic(x):
    try: e_x = math.exp(-x)
    except: e_x = np.exp(-x)
    return 1/(1+e_x)
