
import numpy as np

def data_load(options):
    #%% load the images
    
    Nimages = 100 #100 images, each image has 2 channels

    # load the filenames of all images
    # Note: delete the __MACOSX folder in the img_folder first
    img_filenames = np.array(sorted(os.listdir(options['img_folder'])))#sort to alphabetical order
    assert len(img_filenames)==Nimages*2#2 channels

    wells = [f.split('_')[6] for f in img_filenames]
    wells = np.sort(np.unique(wells))#e.g. A01, A02, ..., E04
    channels = [1,2]
    #%%load the images
    #images, 2 channels
    X = np.zeros(shape=(Nimages,options['target_height'],options['target_width'],2),dtype='float32')
    Y = np.zeros(shape=(Nimages,options['target_height'],options['target_width'],1),dtype='float32')

    i=0
    for w in  wells:
        print('loading image ',i+1)
        for c in channels:
            key = w+'_w'+str(c)
            img_file = None
            for f in img_filenames:
                if key in f:
                    img_file=f
                    break;
            print(img_file)
            #cv2 is better for grayscale images, use 
            #load the image
            img = cv2.imread(options['img_folder']+'/'+img_file,-1)
            #resize
            img=cv2.resize(img,(options['target_width'],options['target_height']))
            #normalize to 0-1
            img=img/img.max()
            X[i,:,:,c-1]=img
        print('loading mask')
        img = cv2.imread(options['msk_folder']+'/'+w+'_binary.png',cv2.IMREAD_GRAYSCALE)
        #resize
        img=cv2.resize(img,(options['target_width'],options['target_height']))
        #normalize to 0-1
        img=img/img.max()
        #create binary image from [0,1] to {0,1}, using 0.5 as threshold
        img[img<0.5]=0
        img[img>=0.5]=1
        Y[i,:,:,0]=img
        i=i+1
        print()#add a blank line for readability

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



    #X_tr = X[ix_tr,:]
    #Y_tr = Y[ix_tr,:]

    X_val = X[ix_val,:]
    Y_val = Y[ix_val,:]

    X_ts = X[ix_ts,:]
    Y_ts = Y[ix_ts,:]

    d={}
    X_tr={}
    Y_tr={}
    #X_val={}
    #Y_val={}
    #X_ts={}
    #Y_ts={}

    for it in range(1,6):
        d['ix_annot'+str(it)], d['ix_unannot'+str(it)] = train_test_split(ix_tr,train_size=0.1*it, random_state=0)
        #d['ix_val'+str(it)], d['ix_unannot'+str(it)] = train_test_split(ix_val,train_size=0.1*it, random_state=0)
        #d['ix_ts'+str(it)], d['ix_unannot'+str(it)] = train_test_split(ix_ts,train_size=0.1*it, random_state=0)
        
        X_tr[str(it)]=X[d['ix_annot'+str(it)],:]
        Y_tr[str(it)]=Y[d['ix_annot'+str(it)],:]
        #X_val[str(it)]=X[d['ix_val'+str(it)],:]
        #Y_val[str(it)]=Y[d['ix_val'+str(it)],:]
        #X_ts[str(it)]=X[d['ix_ts'+str(it)],:]
        #Y_ts[str(it)]=Y[d['ix_ts'+str(it)],:]
        
        print(d['ix_annot'+str(it)])
        #print(d['ix_val'+str(it)])
        #print(d['ix_ts'+str(it)])
        fnames_tr = wells[d['ix_annot'+str(it)]].tolist()
        fnames_val = wells[ix_val].tolist()
        fnames_ts = wells[ix_ts].tolist()

        fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
        df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
                      'split':fname_split})

        #save to disk
        df.to_csv('./data/training_validation_test_splits.csv',index=False)
        np.save('./data/X_tr'+str(it)+'.npy',X_tr[str(it)])
        np.save('./data/X_val.npy',X_val)
        np.save('./data/X_ts.npy',X_ts)

        np.save('./data/Y_tr'+str(it)+'.npy',Y_tr[str(it)])
        np.save('./data/Y_val.npy',Y_val)
        np.save('./data/Y_ts.npy',Y_ts)
