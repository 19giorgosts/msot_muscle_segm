def full_data():
    ## Full data mode to establish a baseline score
    full_data_load()
    train(mc=True)
    p,unc=eval_mc('test')
    X_ts = np.load('./X_ts.npy')
    Y_ts = np.load('./Y_ts.npy')
    perc=100
    unc_plot(X_ts,Y_ts,p,unc,perc)