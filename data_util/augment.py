import numpy as np
import skimage.transform

def rotateT(X,angle):
	#rotate image tensor, TF order, single channel
	X_rot = np.zeros_like(X)
	#repeat for every channel
	for ch in np.arange(X.shape[-1]):
		#print('channel',ch)
		#repeat for every image
		for i in np.arange(X.shape[0]):
			#print('image',i)
			X_rot[i,:,:,ch] = skimage.transform.rotate(X[i,:,:,ch],angle=angle,resize=False,preserve_range=True,mode='edge')
	return(X_rot)

def shiftT(X,dx,dy):
	#rotate image tensor, TF order, single channel
	X_shift = np.zeros_like(X)
	#repeat for every image
	tform = skimage.transform.SimilarityTransform(translation=(dx, dy))
	for i in np.arange(X.shape[0]):
		#print('image',i)
		X_shift[i,:,:,:] = skimage.transform.warp(X[i,:,:,:],tform,mode='edge')
	return(X_shift)

#%%
def aug_generator(X_raw=None,Y_raw=None,
				  batch_size=4,
				  flip_axes=['x','y'],
				  rotation_angles=[5,15]):
				  #noise_gaussian_mean=0,
				  #noise_gaussian_var=1e-2):
				  #noise_snp_amount=0.05):
	
	batch_size=batch_size#recommended batch size    
	Ndatapoints = len(X_raw)
	#Naugmentations=4 #original + flip, rotation, noise_gaussian, noise_snp
	
	while(True):
		#print('start!')
		ix_randomized = np.random.choice(Ndatapoints,size=Ndatapoints,replace=False)
		ix_batches = np.array_split(ix_randomized,int(Ndatapoints/batch_size))
		for b in range(len(ix_batches)):
			#print('step',b,'of',len(ix_batches))
			ix_batch = ix_batches[b]
			current_batch_size=len(ix_batch)
			#print('size of current batch',current_batch_size)
			#print(ix_batch)
			X_batch = X_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
			Y_batch = Y_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
			
			#now do augmentation on images and masks
			#iterate over each image in the batch
			for img in range(current_batch_size):
				#print('current_image',img,': ',ix_batch[img])
				do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
				if do_aug == True:
					#print('flipping',img)
					flip_axis_selected = np.random.choice(flip_axes,1,replace=False)[0]
					if flip_axis_selected == 'x':
						flip_axis_selected = 1
					else: # 'y'
						flip_axis_selected = 0
					#flip an axis
					X_batch[img,:,:,:] = np.flip(X_batch[img,:,:,:],axis=flip_axis_selected)
					Y_batch[img,:,:,:] = np.flip(Y_batch[img,:,:,:],axis=flip_axis_selected)
					#print('Flip on axis',flip_axis_selected)
				
				do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
				if do_aug == True:
					#print('rotating',img)
					rotation_angle_selected = np.random.uniform(low=rotation_angles[0],high=rotation_angles[1],size=1)[0]
					#rotate the image
					X_batch[img,:,:,:] = rotateT(np.expand_dims(X_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
					Y_batch[img,:,:,:] = rotateT(np.expand_dims(Y_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
					#print('Rotate angle',rotation_angle_selected)
			yield(X_batch,Y_batch)
			#print('step end after',b,'of',len(ix_batches))