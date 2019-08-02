
import numpy as np
import matplotlib.pyplot as plt

# here some functions used for plotting purposes

def mask_plt(x,pred,uncertainty,num = 3):
  for i in range(num):
      #sample = np.random.randint(0,len(X_ts[i,:,:,0]))
      #image = X_predict[sample]
      #gt    = Y_predict[sample]
      
      #gt    = np.squeeze(gt)
          
      #n = np.random.randint(0,num)
      fig, ax = plt.subplots(2,2,figsize=(12,6))
      
      
      #fig.suptitle('Dice: {:.2f}'.format(np.median(dice)), y=1.0, fontsize=14)
      
      cax0 = ax[0,0].imshow(x[i,:,:,0])
      plt.colorbar(cax0, ax=ax[0,0])
      ax[0,0].set_title('weight mask 1')
      
      #cax1 = ax[0,1].imshow(segm[i,:,:,0])
      #plt.colorbar(cax1, ax=ax[0,1])
      #ax[0,1].set_title('Max Probability')
      #ax[0,1].set_ylabel('Prediction')
      
      cax2 = ax[0,1].imshow(pred[i,:,:,0])
      plt.colorbar(cax2, ax=ax[0,1])
      ax[0,1].set_title('MC-Dropout')
      ax[0,1].set_ylabel('Prediction')
      
      cax3 = ax[1,0].imshow(x[i,:,:,1])
      plt.colorbar(cax3, ax=ax[1,0])
      ax[1,0].set_title('weight mask 2')
      
      #cax4 = ax[1,1].imshow(max_softmax[i,:,:,0])
      #plt.colorbar(cax4, ax=ax[1,1])
      #ax[1,1].set_xlabel('Max Probability')
      #ax[1,1].set_ylabel('Uncertainty')

      cax5 = ax[1,1].imshow(uncertainty[i,:,:,0])
      plt.colorbar(cax5,ax=ax[1,1])
      #ax[1,1].set_title('MC-Dropout')
      ax[1,1].set_ylabel('Uncertainty')
      
      #for a in ax.flatten(): a.axis('off')
      #for a in ax.flatten(): a.xaxis.set_ticks_position('none')    
      
      fig.savefig('prediction_uncertainty_{:03d}.png'.format(i), dpi=300)
      
      plt.show()
      plt.close()
      plt.clf()

def generate_plot(X,Nmodels,connected = True):
  #%%

  #Ndatapoints: the number of datapoints in the test set
  #Nmodels: the number of models trained

  #matrix containing the data
  #rows: test set datapoints
  #columns: models
  #each elements corresponds to the score (e.g. accuracy) of a model on a dataset
  #X = np.zeros((Ndatapoints,Nmodels),dtype='float')#initialize to zeros

  np.random.seed(1)#set random seed for repeatability

  #%% generate a plot of boxplots whose medians are connected by a line

  figure_size = 4
  fig, ax = plt.subplots(figsize=(figure_size*Nmodels,figure_size))
  box_data = X
  md = np.median(box_data,axis=0)#median
  #ax.boxplot plots each column as a separate boxplot
  bplots = ax.boxplot(box_data)

  xticks = np.arange(Nmodels)+1

  if connected == True:
      #make the boxplots transparent
      for key in bplots.keys():
          #print(key)
          for b in bplots[key]:
              b.set_alpha(0.2)
      #add a line that connects the medians of all boxplots
      ax.plot(xticks,md,marker='o',c='black',lw=5,markersize=15,label='median')
      xlab = '%'
  else:
      xlab = 'Model '

  ax.set_ylabel('Score (test set)',{'fontsize':16})
  ax.set_xlabel('Model',{'fontsize':16})
  ax.set_title('Model Performance ',{'fontsize':16})
  ax.set_ylim(0,1)

  #generate the xtick labels
  xtick_labels = []
  for m in xticks:
      xtick_labels.append(xlab+str(m))
  ax.set_xticklabels(xtick_labels,rotation = 30, ha='center',fontsize=10)

  #save the figure to disk
  if connected == True:
      plt.savefig('model_boxplots_connected.png',dpi=200,bbox_inches='tight')
  else:
      plt.savefig('model_boxplots.png',dpi=200,bbox_inches='tight')

  #%% redo the same plot without the boxplots
  # only plot a line for the medians, along with error-bars for interquantile range

  figure_size = 4
  fig, ax = plt.subplots(figsize=(figure_size*Nmodels,figure_size))
  xticks = np.arange(Nmodels)+1
  md = np.median(box_data,axis=0)
  yerr = np.zeros((2,Nmodels))
  for m in range(Nmodels):
      #plt.errorbar needs the difference between the percentile and the median
      #lower errorbar: 25th percentile
      yerr[0,m]=md[m]-np.percentile(X[:,m],25)#lower errorbar
      #upper errorbar: 75th percentile
      yerr[1,m]=np.percentile(X[:,m],75)-md[m]

  #plot the errorbars
  ax.errorbar(xticks,md,yerr,capsize=10,fmt='none',c='black')
  #fmt='none' to only plot the errorbars

  #plot the (optional) connecting line
  if connected == True:
      ax.plot(xticks,md,marker='o',c='blue',lw=5,markersize=10,label='Random query')
      xlab = '% '
  else:
      ax.scatter(xticks,md,marker='o',c='black',s=200)
      xlab = 'Model '
  plt.axhline(y=0.92, label="Full data performance",color='r', linestyle='--')
  plt.legend()

  ax.set_ylabel('Score (test set)',{'fontsize':16})
  ax.set_xlabel('Model',{'fontsize':16})
  ax.set_title('Model Performance ',{'fontsize':16})
  ax.set_ylim(0,1)

  ax.set_xticks(xticks)
  #generate the xtick labels
  xtick_labels = []
  for m in xticks:
      xtick_labels.append(str(10*m)+xlab)
  ax.set_xticklabels(xtick_labels,rotation = 30, ha='center',fontsize=10)

  if connected == True:
      plt.savefig('model_errorbars_connected.png',dpi=200,bbox_inches='tight')
  else:
      plt.savefig('model_errorbars.png',dpi=200,bbox_inches='tight')