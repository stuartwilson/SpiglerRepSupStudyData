import numpy as np
import pylab as pl
import scipy.io as sio

'''
    EXTRACT DATA
'''

subjs = 21

pathtodata='data/'
pathtoexract='extracted/'
pathtofigs='figs/'

def getContrasts(fileid):
    x = sio.loadmat(pathtodata+'sub01/'+fileid+'.mat')['contrasts']
    for i in range(1,subjs):
        t = ("{:0>2d}".format(i+1))
        y = sio.loadmat(pathtodata+'sub'+t+'/'+fileid+'.mat')['contrasts']
        x = np.hstack([x,y])
    return x.transpose()

def getPSTHs(fileid):
    x = sio.loadmat(pathtodata+'sub01/'+fileid+'.mat')['psth']
    for i in range(1,subjs):
        t = ("{:0>2d}".format(i+1))
        y = sio.loadmat(pathtodata+'sub'+t+'/'+fileid+'.mat')['psth']
        x = np.dstack((x,y))
    return x

# Contrasts (21 ppts x 16 samples)
lFFAc = getContrasts('lFFA_roi_contrasts')
rFFAc = getContrasts('rFFA_roi_contrasts')
lOFAc = getContrasts('lOFA_roi_contrasts')
rOFAc = getContrasts('rOFA_roi_contrasts')

# PSTHs (12 timepoints x 10 samples x 21 ppts)
lFFAp = getPSTHs('lFFA_roi_psth')
rFFAp = getPSTHs('rFFA_roi_psth')
lOFAp = getPSTHs('lOFA_roi_psth')
rOFAp = getPSTHs('rOFA_roi_psth')


'''
    RE-COMBINE DATA
'''

# same trials only
OFAsame1 = np.hstack([lOFAc[:,0],rOFAc[:,0]])
OFAsame2 = np.hstack([lOFAc[:,3],rOFAc[:,3]])
OFAsame3 = np.hstack([lOFAc[:,6],rOFAc[:,6]])

FFAsame1 = np.hstack([lFFAc[:,0],rFFAc[:,0]])
FFAsame2 = np.hstack([lFFAc[:,3],rFFAc[:,3]])
FFAsame3 = np.hstack([lFFAc[:,6],rFFAc[:,6]])

np.savetxt(pathtoexract+'FFAsame1.csv',FFAsame1)
np.savetxt(pathtoexract+'FFAsame2.csv',FFAsame2)
np.savetxt(pathtoexract+'FFAsame3.csv',FFAsame3)

np.savetxt(pathtoexract+'OFAsame1.csv',OFAsame1)
np.savetxt(pathtoexract+'OFAsame2.csv',OFAsame2)
np.savetxt(pathtoexract+'OFAsame3.csv',OFAsame3)

# half trials only
OFAhalf1 = np.hstack([lOFAc[:,2],rOFAc[:,2]])
OFAhalf2 = np.hstack([lOFAc[:,5],rOFAc[:,5]])
OFAhalf3 = np.hstack([lOFAc[:,8],rOFAc[:,8]])

FFAhalf1 = np.hstack([lFFAc[:,2],rFFAc[:,2]])
FFAhalf2 = np.hstack([lFFAc[:,5],rFFAc[:,5]])
FFAhalf3 = np.hstack([lFFAc[:,8],rFFAc[:,8]])

np.savetxt(pathtoexract+'FFAhalf1.csv',FFAhalf1)
np.savetxt(pathtoexract+'FFAhalf2.csv',FFAhalf2)
np.savetxt(pathtoexract+'FFAhalf3.csv',FFAhalf3)

np.savetxt(pathtoexract+'OFAhalf1.csv',OFAhalf1)
np.savetxt(pathtoexract+'OFAhalf2.csv',OFAhalf2)
np.savetxt(pathtoexract+'OFAhalf3.csv',OFAhalf3)


# diff trials only
OFAdiff1 = np.hstack([lOFAc[:,1],rOFAc[:,1]])
OFAdiff2 = np.hstack([lOFAc[:,4],rOFAc[:,4]])
OFAdiff3 = np.hstack([lOFAc[:,7],rOFAc[:,7]])

FFAdiff1 = np.hstack([lFFAc[:,1],rFFAc[:,1]])
FFAdiff2 = np.hstack([lFFAc[:,4],rFFAc[:,4]])
FFAdiff3 = np.hstack([lFFAc[:,7],rFFAc[:,7]])

np.savetxt(pathtoexract+'FFAdiff1.csv',FFAdiff1)
np.savetxt(pathtoexract+'FFAdiff2.csv',FFAdiff2)
np.savetxt(pathtoexract+'FFAdiff3.csv',FFAdiff3)

np.savetxt(pathtoexract+'OFAdiff1.csv',OFAdiff1)
np.savetxt(pathtoexract+'OFAdiff2.csv',OFAdiff2)
np.savetxt(pathtoexract+'OFAdiff3.csv',OFAdiff3)


# Subtract mean beta from condition-specific first faces
rOFAsameRS = rOFAc[:,0]-rOFAc[:,6]
rOFAdiffRS = rOFAc[:,1]-rOFAc[:,7]
rOFAhalfRS = rOFAc[:,2]-rOFAc[:,8]
lOFAsameRS = lOFAc[:,0]-lOFAc[:,6]
lOFAdiffRS = lOFAc[:,1]-lOFAc[:,7]
lOFAhalfRS = lOFAc[:,2]-lOFAc[:,8]
rFFAsameRS = rFFAc[:,0]-rOFAc[:,6]
rFFAdiffRS = rFFAc[:,1]-rOFAc[:,7]
rFFAhalfRS = rFFAc[:,2]-rOFAc[:,8]
lFFAsameRS = lFFAc[:,0]-lOFAc[:,6]
lFFAdiffRS = lFFAc[:,1]-lOFAc[:,7]
lFFAhalfRS = lFFAc[:,2]-lOFAc[:,8]

# Concatenate left and right for each ROI
OFAsameRS = np.hstack([lOFAsameRS,rOFAsameRS])
OFAhalfRS = np.hstack([lOFAhalfRS,rOFAhalfRS])
OFAdiffRS = np.hstack([lOFAdiffRS,rOFAdiffRS])
FFAsameRS = np.hstack([lFFAsameRS,rFFAsameRS])
FFAhalfRS = np.hstack([lFFAhalfRS,rFFAhalfRS])
FFAdiffRS = np.hstack([lFFAdiffRS,rFFAdiffRS])

np.savetxt(pathtoexract+'OFAsameRS.csv',OFAsameRS)
np.savetxt(pathtoexract+'OFAhalfRS.csv',OFAhalfRS)
np.savetxt(pathtoexract+'OFAdiffRS.csv',OFAdiffRS)
np.savetxt(pathtoexract+'FFAsameRS.csv',FFAsameRS)
np.savetxt(pathtoexract+'FFAhalfRS.csv',FFAhalfRS)
np.savetxt(pathtoexract+'FFAdiffRS.csv',FFAdiffRS)


# Subtract mean beta from condition-specific first faces
rOFAsameRS2 = rOFAc[:,0]-rOFAc[:,3]
rOFAdiffRS2 = rOFAc[:,1]-rOFAc[:,4]
rOFAhalfRS2 = rOFAc[:,2]-rOFAc[:,5]
lOFAsameRS2 = lOFAc[:,0]-lOFAc[:,3]
lOFAdiffRS2 = lOFAc[:,1]-lOFAc[:,4]
lOFAhalfRS2 = lOFAc[:,2]-lOFAc[:,5]
rFFAsameRS2 = rFFAc[:,0]-rOFAc[:,3]
rFFAdiffRS2 = rFFAc[:,1]-rOFAc[:,4]
rFFAhalfRS2 = rFFAc[:,2]-rOFAc[:,5]
lFFAsameRS2 = lFFAc[:,0]-lOFAc[:,3]
lFFAdiffRS2 = lFFAc[:,1]-lOFAc[:,4]
lFFAhalfRS2 = lFFAc[:,2]-lOFAc[:,5]

# Concatenate left and right for each ROI
OFAsameRS2 = np.hstack([lOFAsameRS2,rOFAsameRS2])
OFAhalfRS2 = np.hstack([lOFAhalfRS2,rOFAhalfRS2])
OFAdiffRS2 = np.hstack([lOFAdiffRS2,rOFAdiffRS2])
FFAsameRS2 = np.hstack([lFFAsameRS2,rFFAsameRS2])
FFAhalfRS2 = np.hstack([lFFAhalfRS2,rFFAhalfRS2])
FFAdiffRS2 = np.hstack([lFFAdiffRS2,rFFAdiffRS2])

np.savetxt(pathtoexract+'OFAsameRS2.csv',OFAsameRS2)
np.savetxt(pathtoexract+'OFAhalfRS2.csv',OFAhalfRS2)
np.savetxt(pathtoexract+'OFAdiffRS2.csv',OFAdiffRS2)
np.savetxt(pathtoexract+'FFAsameRS2.csv',FFAsameRS2)
np.savetxt(pathtoexract+'FFAhalfRS2.csv',FFAhalfRS2)
np.savetxt(pathtoexract+'FFAdiffRS2.csv',FFAdiffRS2)

# PLOTTING
fs=15

col1 = (0.6,0.6,0.9) # red
col2 = (0.9,0.6,0.6) # blue


'''
    PSTHs - overlay mean PSTH over all first faces and PSTH for second face in same condition
'''

pos = [14,0.25]
F = pl.figure(figsize=(8,8))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
f = F.add_subplot(221)
ax = np.array([-1,23,-0.2,0.3])
dat,tit = lFFAp, 'left FFA'
f.errorbar(np.arange(12)*2,np.mean(np.mean(dat[:,:3,:],axis=1),axis=1),yerr=np.std(np.mean(dat[:,:3,:],axis=1),axis=1)/np.sqrt(subjs),capsize=2,fmt='o-',color='black')
f.errorbar(np.arange(12)*2,np.mean(dat[:,3,:],axis=1),yerr=np.std(dat[:,3,:],axis=1)/np.sqrt(subjs),capsize=2,fmt='s--',color='black')
f.axis(ax)
f.annotate(tit,pos,fontsize=fs)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_ylabel(r'% BOLD change',fontsize=fs)
f.set_xlabel('time ($s$)',fontsize=fs)
f.legend(['1st','2nd'],loc='lower right',frameon=False)
f.text(ax[0]-6,ax[3]+0.02,r'$\bf{A}$',fontsize=fs)

f = F.add_subplot(222)
dat,tit = rFFAp, 'right FFA'
f.errorbar(np.arange(12)*2,np.mean(np.mean(dat[:,:3,:],axis=1),axis=1),yerr=np.std(np.mean(dat[:,:3,:],axis=1),axis=1)/np.sqrt(subjs),capsize=2,fmt='o-',color='black')
f.errorbar(np.arange(12)*2,np.mean(dat[:,3,:],axis=1),yerr=np.std(dat[:,3,:],axis=1)/np.sqrt(subjs),capsize=2,fmt='s--',color='black')
f.axis(ax)
f.annotate(tit,pos,fontsize=fs)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_ylabel(r'% BOLD change',fontsize=fs)
f.set_xlabel('time ($s$)',fontsize=fs)
f.legend(['1st','2nd'],loc='lower right',frameon=False)
f.text(ax[0]-6,ax[3]+0.02,r'$\bf{B}$',fontsize=fs)

f = F.add_subplot(223)
dat,tit = lOFAp, 'left OFA'
f.errorbar(np.arange(12)*2,np.mean(np.mean(dat[:,:3,:],axis=1),axis=1),yerr=np.std(np.mean(dat[:,:3,:],axis=1),axis=1)/np.sqrt(subjs),capsize=2,fmt='o-',color='black')
f.errorbar(np.arange(12)*2,np.mean(dat[:,3,:],axis=1),yerr=np.std(dat[:,3,:],axis=1)/np.sqrt(subjs),capsize=2,fmt='s--',color='black')
f.axis(ax)
f.annotate(tit,pos,fontsize=fs)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_ylabel(r'% BOLD change',fontsize=fs)
f.set_xlabel('time ($s$)',fontsize=fs)
f.legend(['1st','2nd'],loc='lower right',frameon=False)
f.text(ax[0]-6,ax[3]+0.02,r'$\bf{C}$',fontsize=fs)

f = F.add_subplot(224)
dat,tit = rOFAp, 'right OFA'
f.errorbar(np.arange(12)*2,np.mean(np.mean(dat[:,:3,:],axis=1),axis=1),yerr=np.std(np.mean(dat[:,:3,:],axis=1),axis=1)/np.sqrt(subjs),capsize=2,fmt='o-',color='black')
f.errorbar(np.arange(12)*2,np.mean(dat[:,7,:],axis=1),yerr=np.std(dat[:,7,:],axis=1)/np.sqrt(subjs),capsize=2,fmt='s--',color='black')
f.axis(ax)
f.annotate(tit,pos,fontsize=fs)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_ylabel(r'% BOLD change',fontsize=fs)
f.set_xlabel('time ($s$)',fontsize=fs)
f.legend(['1st','2nd'],loc='lower right',frameon=False)
f.text(ax[0]-6,ax[3]+0.02,r'$\bf{D}$',fontsize=fs)

F.tight_layout()

F.savefig(pathtofigs+'PSTHs.pdf')



'''
    SAME TRIALS - combine left and right and show reduction for same trials in each region
'''

N = 2*subjs

minV = 0
maxV = 7.5
wid = 0.75
nbars = 3

# SAME TRIALS ONLY

ax = np.array([-wid, nbars-1+wid, minV, maxV]);
F = pl.figure(figsize=(10,5))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# OFA
h1 = 6.5
h2 = 7

f = F.add_subplot(121)
f.bar(np.arange(nbars),[np.mean(OFAsame1),np.mean(OFAsame2),np.mean(OFAsame3)],yerr=[np.std(OFAsame1),np.std(OFAsame2),np.std(OFAsame3)]/np.sqrt(N),width=wid,color=col1,edgecolor='black',ecolor='black', capsize=5)
f.axis(ax)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.05])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("OFA ('same')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{A}$',fontsize=fs)

# FFA
f = F.add_subplot(122)
f.bar(np.arange(nbars),[np.mean(FFAsame1),np.mean(FFAsame2),np.mean(FFAsame3)],yerr=[np.std(FFAsame1),np.std(FFAsame2),np.std(FFAsame3)]/np.sqrt(N),width=wid,color=col1,edgecolor='black',ecolor='black', capsize=5)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.05])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])

f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("FFA ('same')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{B}$',fontsize=fs)

F.savefig(pathtofigs+'same.pdf')


# HALF TRIALS ONLY

ax = np.array([-wid, nbars-1+wid, minV, maxV]);
F = pl.figure(figsize=(10,5))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# OFA
h1 = 6.5
h2 = 7

f = F.add_subplot(121)
f.bar(np.arange(nbars),[np.mean(OFAhalf1),np.mean(OFAhalf2),np.mean(OFAhalf3)],yerr=[np.std(OFAhalf1),np.std(OFAhalf2),np.std(OFAhalf3)]/np.sqrt(N),width=wid,color=col1,edgecolor='black',ecolor='black', capsize=5)
f.axis(ax)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.05])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("OFA ('half')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{A}$',fontsize=fs)

# FFA
f = F.add_subplot(122)
f.bar(np.arange(nbars),[np.mean(FFAhalf1),np.mean(FFAhalf2),np.mean(FFAhalf3)],yerr=[np.std(FFAhalf1),np.std(FFAhalf2),np.std(FFAhalf3)]/np.sqrt(N),width=wid,color=col2,edgecolor='black',ecolor='black', capsize=5)
f.plot([1,2],[h1,h1],'-|',color='black')
f.annotate('*',[1.5,h1+0.05])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])

f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("FFA ('half')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{B}$',fontsize=fs)



F.savefig(pathtofigs+'half.pdf')


# DIFF TRIALS ONLY

ax = np.array([-wid, nbars-1+wid, minV, maxV]);
F = pl.figure(figsize=(10,5))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# OFA
h1 = 6.5
h2 = 7

f = F.add_subplot(121)
f.bar(np.arange(nbars),[np.mean(OFAdiff1),np.mean(OFAdiff2),np.mean(OFAdiff3)],yerr=[np.std(OFAdiff1),np.std(OFAdiff2),np.std(OFAdiff3)]/np.sqrt(N),width=wid,color=col2,edgecolor='black',ecolor='black', capsize=5)
f.axis(ax)
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("OFA ('diff.')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{A}$',fontsize=fs)

# FFA
f = F.add_subplot(122)
f.bar(np.arange(nbars),[np.mean(FFAdiff1),np.mean(FFAdiff2),np.mean(FFAdiff3)],yerr=[np.std(FFAdiff1),np.std(FFAdiff2),np.std(FFAdiff3)]/np.sqrt(N),width=wid,color=col2,edgecolor='black',ecolor='black', capsize=5)
f.plot([1,2],[h1,h1],'-|',color='black')
f.annotate('*',[1.5,h1+0.05])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.05])

f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(8))
f.set_xticklabels(['1st','2nd','3rd'])
f.set_title("FFA ('diff.')",fontsize=fs)
f.set_ylabel(r'$\beta$',fontsize=fs)
f.set_xlabel('triplet position',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{B}$',fontsize=fs)

F.savefig(pathtofigs+'diff.pdf')

'''
    MAIN ANALYSIS
'''


maxV = 3.5
nbars = 3
ax = np.array([-wid, nbars-1+wid, minV, maxV]);
h1 = 3
h2 = 3.2

F = pl.figure(figsize=(10,5))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# OFA
f = F.add_subplot(121)
f.bar(np.arange(nbars),[np.mean(OFAsameRS),np.mean(OFAhalfRS),np.mean(OFAdiffRS)],yerr=[np.std(OFAsameRS),np.std(OFAhalfRS),np.std(OFAdiffRS)]/np.sqrt(N),width=wid,color=(0.7,0.7,0.7),edgecolor='black',ecolor='black', capsize=5)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.025])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.025])
f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(4))
f.set_xticklabels(["100%\n'same'","50%\n'half'","0%\n'diff.'"])
f.set_title('OFA',fontsize=fs)
f.set_ylabel(r'repetition suppression index',fontsize=fs)# $\left(\beta_1-\beta_3\right)$',fontsize=fs)
f.set_xlabel('similarity of second in triplet',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{A}$',fontsize=fs)

# FFA
f = F.add_subplot(122)
f.bar(np.arange(nbars),[np.mean(FFAsameRS),np.mean(FFAhalfRS),np.mean(FFAdiffRS)],yerr=[np.std(FFAsameRS),np.std(FFAhalfRS),np.std(FFAdiffRS)]/np.sqrt(N),width=wid,color=(0.7,0.7,0.7),edgecolor='black',ecolor='black', capsize=5)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.025])
f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(4))
f.set_xticklabels(["100%\n'same'","50%\n'half'","0%\n'diff.'"])
f.set_title('FFA',fontsize=fs)
f.set_ylabel(r'repetition suppression index',fontsize=fs)# $\left(\beta_1-\beta_3\right)$',fontsize=fs)
f.set_xlabel('similarity of second in triplet',fontsize=fs)
f.text(ax[0]-0.5,ax[3]+0.5,r'$\bf{B}$',fontsize=fs)

F.savefig(pathtofigs+'maineffect.pdf')


'''
    SECOND
'''

F = pl.figure(figsize=(10,5))
F.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# OFA
f = F.add_subplot(121)
f.bar(np.arange(nbars),[np.mean(OFAsameRS2),np.mean(OFAhalfRS2),np.mean(OFAdiffRS2)],yerr=[np.std(OFAsameRS2),np.std(OFAhalfRS2),np.std(OFAdiffRS2)]/np.sqrt(N),width=wid,color=(0.7,0.7,0.7),edgecolor='black',ecolor='black', capsize=5)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.025])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.025])
f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(4))
f.set_xticklabels(["100%\n'same'","50%\n'half'","0%\n'diff.'"])
f.set_title('OFA',fontsize=fs)
f.set_ylabel(r'attenuation',fontsize=fs)
f.set_xlabel('similarity of second in triplet',fontsize=fs)

# FFA
f = F.add_subplot(122)
f.bar(np.arange(nbars),[np.mean(FFAsameRS2),np.mean(FFAhalfRS2),np.mean(FFAdiffRS2)],yerr=[np.std(FFAsameRS2),np.std(FFAhalfRS2),np.std(FFAdiffRS2)]/np.sqrt(N),width=wid,color=(0.7,0.7,0.7),edgecolor='black',ecolor='black', capsize=5)
f.plot([0,1],[h1,h1],'-|',color='black')
f.annotate('*',[0.5,h1+0.025])
f.plot([0,2],[h2,h2],'-|',color='black')
f.annotate('*',[1,h2+0.025])
f.axis(ax)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks(np.arange(nbars))
f.set_yticks(np.arange(4))
f.set_xticklabels(["100%\n'same'","50%\n'half'","0%\n'diff.'"])
f.set_title('FFA',fontsize=fs)
f.set_ylabel(r'attenuation',fontsize=fs)
f.set_xlabel('similarity of second in triplet',fontsize=fs)


F.savefig(pathtofigs+'second.pdf')
