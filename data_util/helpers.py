
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



def train(mode,mc=False):
      
    from keras import backend as K
    K.clear_session()
    model={}
    #load the data from already split files
    X_tr = np.load('./data/X_tr'+mode+'.npy')
    Y_tr = np.load('./data/Y_tr'+mode+'.npy')
    print('Training with '+str(X_tr.shape[0])+' images')
    X_val = np.load('./data/X_val.npy')
    Y_val = np.load('./data/Y_val.npy')
    print('Validating with '+str(X_val.shape[0])+' images')
    #%% train the model
    filepath = 'unet_div8_495K'

    #save the model when val_loss improves during training
    checkpoint = ModelCheckpoint('./trained_models/'+filepath+'_'+mode+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #save training progress in a .csv
    csvlog = CSVLogger('./trained_models/'+filepath+'_'+mode+'_train_log.csv',append=True)
    #stop training if no improvement has been seen on val_loss for a while
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8)
    batch_size=3
    
    #initialize the generator
    gen_train = aug_generator(X_tr,Y_tr,batch_size=batch_size,flip_axes=[1,2])
    #split the array and see how many splits there are to determine #steps
    steps_per_epoch_tr = len(np.array_split(np.zeros(len(X_tr)),int(len(X_tr)/batch_size)))
    
    ## Step 1: Training UNET
    
    # Setup the model
    if (mc==False): 
        print("Training simple Unet")
        model=UNET(X_tr)
    else:
        print("Training Unet-mcdropout")
        model=UNET_mc(X_tr)
        
    #actually do the training
    model.fit_generator(gen_train,
                          steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                          validation_data=(X_val,Y_val),
                          epochs=80,
                          verbose=2,
                          initial_epoch=0,
                          callbacks=[checkpoint, csvlog, early_stopping])
    
    if mc==True:
        
        predictions,uncertainty_mc=uncertainty_calc(X_tr,model,1,sample_times=20)
        val_predictions,val_uncertainty_mc=uncertainty_calc(X_val,model,1,sample_times=20)
        ## Step 2: Training DiceNet
        
        merged=np.concatenate((X_tr,predictions,uncertainty_mc),axis=-1)
        val_merged=np.concatenate((X_val,val_predictions,val_uncertainty_mc),axis=-1)

        model_dice=DiceNet(merged)
        #save the model when val_loss improves during training
        checkpoint = ModelCheckpoint('./trained_models/'+filepath+'_'+mode+'DiceNet.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        #save training progress in a .csv
        csvlog = CSVLogger('./trained_models/'+filepath+'_'+mode+'DiceNet_train_log.csv',append=True)
        #%% calculate dice
        dice = []
        dice_val=[]
        N=len(X_tr)
        for i in range(N):
            dice.append(dice2D(Y_tr[i,:,:,0],predictions[i,:,:,0]))
        #gen_train_diceNet = aug_generator(merged,np.median(dice),batch_size=batch_size,flip_axes=[1,2])
        d=np.array(dice)
        d = d.reshape(-1, 1)
        print(d.shape)
        N=len(X_val)
        for i in range(N):
            dice_val.append(dice2D(Y_val[i,:,:,0],val_predictions[i,:,:,0]))
        #gen_train_diceNet = aug_generator(merged,np.median(dice),batch_size=batch_size,flip_axes=[1,2])
        d_val=np.array(dice_val)
        d_val = d_val.reshape(-1, 1)
        model_dice.fit(merged,d,
                          #steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                          validation_data=(val_merged,dice_val),
                          epochs=60,
                          verbose=2,
                          initial_epoch=0,
                          callbacks=[checkpoint, csvlog, early_stopping])
        del model_dice
        #print(X_tr.shape,predictions.shape,uncertainty_mc.shape)
    del model


  
def eval(mode):
    X_ts = np.load('./data/X_ts.npy')
    Y_ts = np.load('./data/Y_ts.npy')#.astype('float32')#to match keras predicted mask

    Ntest=len(X_ts)

    df = pd.read_csv('./data/training_validation_test_splits.csv')
    well_ts = df[df['split']=='test']['well'].tolist()
    #Y_ts is a binary mask
    #np.unique(Y_ts)
    #array([ 0.,  1.], dtype=float32)

    #%% get predicted masks for test set
    model = load_model('./trained_models/unet_div8_495K'+'_'+mode+'.hdf5')
    print("Loading model: "+'./trained_models/unet_div8_495K'+'_'+mode+'.hdf5')
    
    start = time.time()
    Y_ts_hat = model.predict(X_ts,batch_size=1)
    #print(Y_ts_hat.shape)
    #max_softmax=softmaxEquation(p_hat)
    end = time.time()
    chkpnt=(end-start) # this is how long it lasts for all images of the test set, for Maximum Softmax Probability-based Uncertainty Estimation
    print("\n Time needed for Maximum Softmax Probability-based Uncertainty Estimation  ",chkpnt)

    #%% convert predicted mask to binary
    threshold=0.5
    Y_ts_hat[Y_ts_hat<threshold]=0
    Y_ts_hat[Y_ts_hat>=threshold]=1
     
    #%% calculate dice
    dice = []
    for i in range(Ntest):
        dice.append(dice2D(Y_ts[i,:,:,0],Y_ts_hat[i,:,:,0]))
    #dice = np.array(dice)
    return Y_ts_hat,dice

def eval_mc(mode,sample_times=20):
    X_ts = np.load('./data/X_ts.npy')
    Y_ts = np.load('./data/Y_ts.npy')#.astype('float32')#to match keras predicted mask
    print(X_ts.shape,Y_ts.shape)
    batch_size=1
    Ntest=len(X_ts)
    df = pd.read_csv('./data/training_validation_test_splits.csv')
    well_ts = df[df['split']=='test']['well'].tolist()
  
    #%% get predicted masks for test set
    model = load_model('./trained_models/unet_div8_495K'+'_'+mode+'.hdf5')
    print("Loading model: "+'./trained_models/unet_div8_495K'+'_'+mode+'.hdf5')
 
    pred,uncertainty_mc=uncertainty_calc(X_ts,model,batch_size,sample_times=20)
    return pred,uncertainty_mc

def eval_DiceNet(mode):
    X_ts = np.load('./data/X_ts.npy')
    Y_ts = np.load('./data/Y_ts.npy')#.astype('float32')#to match keras predicted mask

    Ntest=len(X_ts)

    #%% get predicted masks for test set
    model = load_model('./trained_models/unet_div8_495K'+'_'+mode+'DiceNet.hdf5')
    print("Loading model: "+'./trained_models/unet_div8_495K'+'_'+mode+'DiceNet.hdf5')
    print(X_ts.shape)
    ts_predictions,ts_uncertainty_mc=uncertainty_calc(X_ts,model,1,sample_times=20)
    ts_merged=np.concatenate((X_ts,ts_predictions,ts_uncertainty_mc),axis=-1)
    print(ts_merged.shape)
    qest = model.predict(ts_merged,batch_size=1)
    return qest
