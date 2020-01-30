import numpy as np
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
        
        #%% convert predictedυγιείς mask to binary
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

	Ndatapoints = 10 #number of datapoints in the test set
	dice={}
	X_rand = np.zeros((Ndatapoints*Nmodels,Ndata),dtype='float') #initialize to zeros
	X_sg = np.zeros((Ndatapoints*Nmodels,Ndata),dtype='float') #initialize to zeros

	for d in range(1,Ndata+1):
	    for m in range(Ndatapoints*Nmodels):
	        X_rand[m,d-1] = dice_rand['%s'%d][m]
	for d in range(1,Ndata+1):
	    for m in range(Ndatapoints*Nmodels):
	        X_sg[m,d-1] = dice_sg['%s'%d][m]
	return X_rand,X_sg
def sample_prediction(dc,sample_size,random=False):
    dc=np.asarray(dc, dtype=np.float32)
    dc = dc.ravel()
    df = pd.read_csv('./training_pool.csv')
    sug_annot=[]
    if random==True: # receives random samples and feeds them to the RANDOM - annotation procedure 
        #while len(sug_annot)<6:
        #idx=r.randint(0,df.shape[0])
        #if ((df['idx']==idx).any())==True:
        #    indices=df['idx'].iloc[idx]
        #    sug_annot.append(indices)
        idx = np.argpartition(dc, sample_size)
        idx=r.sample(list(idx),sample_size)
        indices=df['idx'].loc[idx]
        for i in range (len(indices)):
                     sug_annot.append(indices.iloc[i])
    else:     # predicts worse n (n=sample_size) samples in the X_pool set, then appends their indices in a list for future annotation
        idx = np.argpartition(dc, -sample_size)[-sample_size:]
        indices=df['idx'].loc[idx]
        for i in range (indices.shape[0]):
             sug_annot.append(indices.iloc[i])
    return sug_annot
def dice_pred(p,Y_ts):
    d=[]
    for i in range(Y_ts.shape[0]):
    #plt.imshow(Y_ts[0,:,:,0])
    #plt.show()
    #plt.imshow(p[0,:,:,0])
    #plt.show()
    #plt.imshow(unc[0,:,:,0])
    #plt.show()
        d.append((dice2D(p[i,:,:,0],Y_ts[i,:,:,0])))
    return d