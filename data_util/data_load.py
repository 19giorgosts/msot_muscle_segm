import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def full_data_load():
    # load the filenames of all images
    img_filenames = np.array(sorted(os.listdir(data)))#sort to alphabetical order
    assert len(img_filenames)==Nimages
    wells = [f.split('.mat')[0] for f in img_filenames]
    wells = np.sort(np.unique(wells))
    #channels = np.arange(1,29) ## full channel approach
    channels = np.array([6,11,16,24,28])
    #%%load the images
    X = np.zeros(shape=(Nimages,target_height,target_width,5),dtype='float32')
    Y = np.zeros(shape=(Nimages,target_height,target_width,1),dtype='float32')

    i=0
    for w in  wells:
        print('loading image ',i)
        j=0
        for c in channels:
            #print(c)
            key = w
            img_file = None
            for f in img_filenames:
                if key in f:
                    img_file=f
                    break;
            #load the image
            img = scipy.io.loadmat(data+'/'+img_file)['Recons'][:,:,c-1]
            #normalize to 0-1
            img=img/img.max()
            X[i,:,:,j]=img
            j=j+1
        img = scipy.io.loadmat(data+'/'+w)['roimask']
        #normalize to 0-1
        img=img/img.max()
        #create binary image from [0,1] to {0,1}, using 0.5 as threshold
        img[img<0.5]=0
        img[img>=0.5]=1
        Y[i,:,:,0]=img
        i=i+1
    #double-check that the masks are binary
    assert np.array_equal(np.unique(Y), [0,1])

    #%% split into train, validation and test sets

    ix = np.arange(len(wells))

    ix_tr, ix_val_ts = train_test_split(ix,train_size=60, random_state=0)
    ix_val, ix_ts = train_test_split(ix_val_ts,train_size=20, random_state=0)

    #sanity check, no overlap between train, validation and test sets
    assert len(np.intersect1d(ix_tr,ix_val))==0
    assert len(np.intersect1d(ix_tr,ix_ts))==0
    assert len(np.intersect1d(ix_val,ix_ts))==0
    print(ix_tr.shape,ix_val.shape,ix_ts.shape)
    
    X_tr = X[ix_tr,:]
    Y_tr = Y[ix_tr,:]
    print(X_tr.shape,Y_tr.shape)
    X_val = X[ix_val,:]
    Y_val = Y[ix_val,:]
    print(X_val.shape,Y_val.shape)
    X_ts = X[ix_ts,:]
    Y_ts = Y[ix_ts,:]
    print(X_ts.shape,Y_ts.shape)
    fnames_tr = wells[ix_tr].tolist()
    fnames_val = wells[ix_val].tolist()
    fnames_ts = wells[ix_ts].tolist()

    fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
    df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
                  'split':fname_split})

    #save to disk
    df.to_csv('./training_validation_test_splits.csv',index=False)

    np.save('./X_tr.npy',X_tr)
    np.save('./X_val.npy',X_val)
    np.save('./X_ts.npy',X_ts)
    np.save('./Υ_tr.npy',Y_tr)
    np.save('./Y_val.npy',Y_val)
    np.save('./Y_ts.npy',Y_ts)

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
def data_load(sg):
    # load the filenames of all images
    img_filenames = np.array(sorted(os.listdir(data)))#sort to alphabetical order
    assert len(img_filenames)==Nimages
    wells = [f.split('.mat')[0] for f in img_filenames]
    wells = np.sort(np.unique(wells))
    channels = np.arange(1,29) # full channel approach
    #channels = np.array([6,11,16,24,28]) #top5 channel approach 
    #%%load the images
    X = np.zeros(shape=(Nimages,target_height,target_width,28),dtype='float32')
    Y = np.zeros(shape=(Nimages,target_height,target_width,1),dtype='float32')

    i=0
    for w in  wells:
        print('loading image ',i)
        j=0
        for c in channels:
            key = w
            img_file = None
            for f in img_filenames:
                if key in f:
                    img_file=f
                    break;
            #cv2 is better for grayscale images, use 
            #load the image
            img = scipy.io.loadmat(data+'/'+img_file)['Recons'][:,:,c-1]
            #print(img.shape)
            #resize
            #img=cv2.resize(img,(target_width,target_height))
            #normalize to 0-1
            img=img/img.max()
            X[i,:,:,c-1]=img
            #j=j+1 #top5 channel approach 
        #print('loading mask')
        img = scipy.io.loadmat(data+'/'+w)['roimask']
        #resize
        #img=cv2.resize(img,(target_width,target_height))
        #normalize to 0-1
        img=img/img.max()
        #create binary image from [0,1] to {0,1}, using 0.5 as threshold
        img[img<0.5]=0
        img[img>=0.5]=1
        Y[i,:,:,0]=img
        i=i+1
        #print()#add a blank line for readability

    #double-check that the masks are binary
    assert np.array_equal(np.unique(Y), [0,1])


    #%% split into train, validation and test sets

    ix = np.arange(len(wells))

    ix_tr, ix_val_ts = train_test_split(ix,train_size=80, random_state=0)
    ix_val, ix_ts = train_test_split(ix_val_ts,train_size=10, random_state=0)

    #sanity check, no overlap between train, validation and test sets
    assert len(np.intersect1d(ix_tr,ix_val))==0
    assert len(np.intersect1d(ix_tr,ix_ts))==0
    assert len(np.intersect1d(ix_val,ix_ts))==0
    print(ix_tr.shape,ix_val.shape,ix_ts.shape)
    X_val = X[ix_val,:]
    Y_val = Y[ix_val,:]

    X_ts = X[ix_ts,:]
    Y_ts = Y[ix_ts,:]
    
    ## HERE we initially start with 10% of images only ! suggestive-annotation mode
    if len(sg)==0:
        ix_annot, ix_unannot = train_test_split(ix_tr,train_size=0.25, random_state=0) # here splitting the set into annotated and unannotated training images (to be annotated subsequently)
    if len(sg)!=0:
        #initialization using .csv column 'idx'
        df_temp=pd.read_csv('./training_validation_test_splits.csv')
        ix_annot=df_temp[df_temp['split']=='train']['idx']
        df_temp1=pd.read_csv('./training_pool.csv')
        ix_unannot=np.array(df_temp1[df_temp1['split']=='train']['idx'])     
        
        for j in sg:
            ix_annot=np.append(ix_annot,j)
            #print(ix_annot)
            ix_unannot = np.delete(ix_unannot, np.where(ix_unannot == j))
            #print(ix_unannot)
    print(len(ix_annot))
    print(len(ix_unannot))

    X_tr=X[ix_annot,:]
    Y_tr=Y[ix_annot,:]

    X_pool=X[ix_unannot,:]
    Y_pool=Y[ix_unannot,:]

    fnames_tr = wells[ix_annot].tolist()
    fnames_pool = wells[ix_unannot].tolist()
    fnames_val = wells[ix_val].tolist()
    fnames_ts = wells[ix_ts].tolist()

    fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
    df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
                  'split':fname_split,
                  'idx': np.append(np.append(ix_annot,ix_val),ix_ts)})

    df1=pd.DataFrame({'well':fnames_pool,'split':'train','idx':ix_unannot})


    #save to disk
    df.to_csv('./training_validation_test_splits.csv',index=False)
    df1.to_csv('./training_pool.csv',index=False)

    np.save('./X_tr.npy',X_tr)
    np.save('./X_pool.npy',X_pool)
    np.save('./X_val.npy',X_val)
    np.save('./X_ts.npy',X_ts)

    np.save('./Y_tr.npy',Y_tr)
    np.save('./Y_pool.npy',Y_pool)
    np.save('./Y_val.npy',Y_val)
    np.save('./Y_ts.npy',Y_ts)

    #%% set-up the UNET model