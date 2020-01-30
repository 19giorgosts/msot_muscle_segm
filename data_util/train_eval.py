
import numpy as np
import configparser,os
from data_util.load_options import *
from data_util.data_load import *
from data_util.helpers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from data_util.augment import *
from networks import *


CURRENT_PATH = os.getcwd()
user_config = configparser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)
data_load(options)


def train(mc=False):
    
    from keras import backend as K
    K.clear_session()
    model={}
    #load the data from already split files
    X_tr = np.load(options['data_folder']+'X_tr'+mode+'.npy')
    Y_tr = np.load(options['data_folder']+'Y_tr'+mode+'.npy')
    print('START TRAINING......:')
    print('Training with '+str(X_tr.shape[0])+' images')
    X_val = np.load(options['data_folder']+'X_val.npy')
    Y_val = np.load(options['data_folder']+'Y_val.npy')
    print('Validating with '+str(X_val.shape[0])+' images')
    #%% train the model
    filepath = 'unet_div8_495K_'

    #save the model when val_loss improves during training
    #hFile = h5py.File('./trained_models/'+filepath+'_'+mode+'.hdf5')
    checkpoint = ModelCheckpoint(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    #save training progress in a .csv
    csvlog = CSVLogger(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'_train_log.csv',append=True)
    #stop training if no improvement has been seen on val_loss for a while
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12)
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
        print(X_tr.shape)
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
        #f = h5py.File('./trained_models/'+filepath+'_'+mode+'DiceNet.hdf5')
        checkpoint = ModelCheckpoint(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'_DiceNet.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        #f.close()
        #save training progress in a .csv
        csvlog = CSVLogger(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'_DiceNet_train_log.csv',append=True)
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
        #d_val=np.array(dice_val)
        #d_val = d_val.reshape(-1, 1)
        model_dice.fit(merged,d,
                          #steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                          validation_data=(val_merged,dice_val),
                          epochs=60,
                          verbose=2,
                          initial_epoch=0,
                          callbacks=[checkpoint, csvlog, early_stopping])
        del model_dice
        hFile = h5py.File(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'.hdf5','r')
        hFile.close()
        print(hFile.__bool__())
        hFile = h5py.File(options['root_folder']+'/trained_models/'+filepath+str(X_tr.shape[0])+'_DiceNet.hdf5','r')
        hFile.close()
        print(hFile.__bool__())
        #print(X_tr.shape,predictions.shape,uncertainty_mc.shape)
    del model
    #print(hFile.__bool__())


def eval(mode):  ## evaluation function for simple UNet
    if mode=='pool':
        X = np.load('/X_pool.npy')
        Y = np.load('./Y_pool.npy')#.astype('float32')#to match keras predicted mask
    elif mode=='test':
        X = np.load('./X_ts.npy')
        Y = np.load('./Y_ts.npy')
    X_tr = np.load('./X_tr.npy') # just to set the correct name based on the size
    print('START EVALUATION......:')
    N=len(X)

    df = pd.read_csv('./training_validation_test_splits.csv')
    well_ts = df[df['split']=='test']['well'].tolist()
    #Y_ts is a binary mask
    #np.unique(Y_ts)
    #array([ 0.,  1.], dtype=float32)

    #%% get predicted masks for test set
    model = load_model('./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'.hdf5')
    print("Loading model: "+'./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'.hdf5')
    
    start = time.time()
    Y_hat = model.predict(X,batch_size=1)
    #print(Y_ts_hat.shape)
    #max_softmax=softmaxEquation(p_hat)
    end = time.time()
    chkpnt=(end-start) # this is how long it lasts for all images of the test set, for Maximum Softmax Probability-based Uncertainty Estimation
    print("\n Time needed for Maximum Softmax Probability-based Uncertainty Estimation  ",chkpnt)

    #%% convert predicted mask to binary
    threshold=0.5
    Y_hat[Y_hat<threshold]=0
    Y_hat[Y_hat>=threshold]=1
     
    #%% calculate dice
    dice = []
    for i in range(N):
        dice.append(dice2D(Y[i,:,:,0],Y_hat[i,:,:,0]))
    #dice = np.array(dice)
    return Y_hat,dice


def eval_mc(mode,sample_times=5): # evaluation function for MC Dropout UNet
    print('START EVALUATION......:')
    if mode=='pool':
        X = np.load('./X_pool.npy')
    elif mode=='test':
        X = np.load('./X_ts.npy')
        
    X_tr = np.load('./X_tr.npy')  # just to set the correct name based on the size

    print(X.shape)
    Ntest=len(X)
    df = pd.read_csv('./training_validation_test_splits.csv')
    well_ts = df[df['split']=='test']['well'].tolist()
  
    #%% get predicted masks for test set
    print("Loading model: "+'./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'.hdf5')
    f = h5py.File('./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'.hdf5')
    f.close()
    model = load_model('./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'.hdf5')
    #print(f.__bool__())
    
    pred,uncertainty_mc=uncertainty_calc(X,model,1,sample_times)
    return pred,uncertainty_mc

    
def eval_DiceNet(mode):  # evaluation function for the DiceNet network
    if mode=='pool':
        X = np.load('./X_pool.npy')
    elif mode=='test':
        X = np.load('./X_ts.npy')
        
    X_tr = np.load('./X_tr.npy')  # just to set the correct name based on the size

    #%% get predicted masks for test set
    print("Loading model: "+'./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'_DiceNet.hdf5')
    #f = h5py.File('./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'_DiceNet.hdf5')
    #f.close()
    model = load_model('./trained_models/unet_div8_495K_'+str(X_tr.shape[0])+'_DiceNet.hdf5')
    #print(f.__bool__())
    p,uncertainty_mc = eval_mc(mode)
    merged=np.concatenate((X,p,uncertainty_mc),axis=-1)
    qest=model.predict(merged)
    return qest