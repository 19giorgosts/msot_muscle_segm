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



def run_random(Nmodels,Ndata,Ndatapoints): ## RANDOM ANNOTATION PROCEDURE
    #Nmodels = 5 #number of models trained
    #Ndata= 5 #number of modes(percentages%)
    #Ndatapoints = 10 #number of datapoints in the test set
    dice={}
    import json
    for mode in range(1,Ndata+1):
        print("Running mode "+str(mode))
        X = np.zeros((Ndatapoints*Nmodels,Ndata),dtype='float') #initialize to zeros
        l=[]
        for i in range (Nmodels):
            if mode==1: sug_annot=[]## running mode 10%
            if i==0: data_load(sug_annot) # data loading
            X_ts = np.load('./X_ts.npy')
            Y_ts = np.load('./Y_ts.npy')
            train(mc=True) # training procedure ( whole network )dicehat=annot_procedure(sug_annot) #training and evaluation of whole pipeline for a particular % of train split
            p,unc=eval_mc(mode='test') #evaluation on test set for model performance
            dc_rand=dice_pred(p,Y_ts)
            unc_plot(X_ts,Y_ts,p,unc,mode*25)
            dicehat=eval_DiceNet(mode='pool') # evaluation on pool set to get the estimated dice scores
            for i in dc_rand:
                l.append(i)
                dice[mode]=l
        sug_annot = sample_prediction(dicehat,sample_size=20,random=True) # random annotation
        with open('dice_msot_rand.json', 'w') as fp:
            json.dump(dice, fp)



def run_sug(Nmodels,Ndata,Ndatapoints):## SUGGESTIVE ANNOTATION 
    #Nmodels = 5 #number of models trained
    #Ndata= 5 #number of modes(percentages%)
    #Ndatapoints = 10 #number of datapoints in the test set
    dice={}
    import json
    for mode in range(1,Ndata+1):
        print("Running mode "+str(mode))
        X = np.zeros((Ndatapoints*Nmodels,Ndata),dtype='float') #initialize to zeros
        l=[]
        for i in range (Nmodels):
            if mode==1: sug_annot=[]## running mode 10%
            if i==0: data_load(sug_annot) # data loading
            X_ts = np.load('./X_ts.npy')
            Y_ts = np.load('./Y_ts.npy')
            train(mc=True) # training procedure ( whole network )dicehat=annot_procedure(sug_annot) #training and evaluation of whole pipeline for a particular % of train split
            p,unc=eval_mc(mode='test') #evaluation on test set for model performance
            dc_rand=dice_pred(p,Y_ts)
            unc_plot(X_ts,Y_ts,p,unc,mode*25)
            dicehat=eval_DiceNet(mode='pool') # evaluation on pool set to get the estimated dice scores
            for i in dc_rand:
                l.append(i)
                dice[mode]=l
        sug_annot = sample_prediction(dicehat,sample_size=20) # random annotation
        with open('dice_msot_sg.json', 'w') as fp:
            json.dump(dice, fp)


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