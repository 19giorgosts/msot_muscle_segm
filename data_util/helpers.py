
import numpy as np

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def uncertainty_calc(X,model,bz,sample_times):
    #MC Dropout implementation
    
    for batch_id in tqdm(range(X.shape[0] // bz)):
        # initialize our predictions
        Y_hat = np.zeros(shape=(sample_times,X.shape[0],target_height,target_width,1),dtype='float32')
        #print(Y_ts_hat.shape) 
        
        start = time.time() # MC dropout is starting !!
        
        for sample_id in range(sample_times):
            # predict stochastic dropout model T times
            Y_hat[sample_id] = model.predict(X, bz)
        # average over all passes
        prediction = Y_hat.mean(axis=0)       
        uncertainty_mc= -(prediction*np.log2(prediction) + (1-prediction)*np.log2(1-prediction)) ## entropy = -(p*np.log2(p) + (1-p)*np.log(1-p))
        end = time.time()        
        chkpnt=(end-start) # this is how long it lasts for all images of the test set, for MC Dropout-based Uncertainty Estimation
        
        #%% convert predicted mask to binary
        threshold=0.5
        prediction[prediction<threshold]=0
        prediction[prediction>=threshold]=1
        print("\n Time needed for MC Dropout-based Uncertainty Estimation  ",chkpnt)

    return prediction,uncertainty_mc

def dice2D(a,b):
    intersection = np.sum(a[b==1])
    dice = (2*intersection)/(np.sum(a)+np.sum(b))
    if (np.sum(a)+np.sum(b))==0: #black/empty masks
        dice=1.0
    return(dice)

def parsing(Nmodels,Ndata):
#Nmodels = 1 #number of models trained
#Ndata=2 #number of modes(percentages%)

  dice={}
  for mode in range(1,Ndata+1):
      print("Running mode "+str(mode))
      X_ts = np.load(options['data_folder']+'X_ts'+str(mode)+'.npy')
      Ndatapoints = len(X_ts) #number of datapoints in the test set e.g. 2(mode1),4(mode2).. etc.
      X = np.zeros((Ndatapoints*Nmodels,Ndata),dtype='float')#initialize to zeros
      l=[]
      for i in range (Nmodels):
          train(str(mode))
          ev,dc=eval(str(mode))
          for i in dc:
              l.append(i)
          dice[mode]=l
      for d in range(mode,len(dice)+1):
          for m in range(Ndatapoints*Nmodels):
              print(d,m)
              X[m,d] = dice[d][m]
  return X