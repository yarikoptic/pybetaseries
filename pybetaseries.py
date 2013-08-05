#!/usr/bin/env python
"""pybetaseries: a module for computing beta-series regression on fMRI data

Includes:
pybetaseries: main function
estimate_OLS: helper function to estimate least squares model
spm_hrf: helper function to generate double-gamma HRF
"""

#from mvpa.misc.fsl.base import *
from mvpa2.misc.fsl.base import *
from mvpa2.datasets.mri import fmri_dataset

import numpy as N
import nibabel
import scipy.stats
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
#from mvpa.datasets.mri import *
import os
from copy import copy

from mvpa2.base import verbose

def get_smoothing_kernel(cutoff, ntp):
    sigN2 = (cutoff/(N.sqrt(2.0)))**2.0
    K = toeplitz(1
                 /N.sqrt(2.0*N.pi*sigN2)
                 *N.exp((-1*N.array(range(ntp))**2.0/(2*sigN2))))
    K = spdiags(1./N.sum(K.T, 0).T, 0, ntp, ntp)*K
    H = N.zeros((ntp, ntp)) # Smoothing matrix, s.t. H*y is smooth line
    X = N.hstack((N.ones((ntp, 1)), N.arange(1, ntp+1).T[:, N.newaxis]))
    for  k in range(ntp):
        W = N.diag(K[k, :])
        Hat = N.dot(N.dot(X, N.linalg.pinv(N.dot(W, X))), W)
        H[k, :] = Hat[k, :]

    F = N.eye(ntp) - H
    return F

def pybetaseries(fsfdir,
                 methods=['lsall', 'lsone'],
                 time_res=0.1,
                 modeldir=None,
                 outdir=None,
                 designdir=None,
                 design_fsf_file='design.fsf',
                 design_mat_file='design.mat',
                 data_file=None,
                 mask_file=None,
                 collapse_other_conditions=True):
    """Compute beta-series regression on a feat directory

    Required arguments:

    fsfdir: full path of a feat directory

    Optional arguments:

    method: list of methods to be used, can include:
    'lsone': single-trial iterative least squares estimation from Turner & Ashby
    'lsall': standard beta-series regression from Rissman et al.

    time_res: time resolution of the model used to generate the convolved design matrix

    outdir: where to store the results
    designdir: location of design_mat_file (e.g. design.mat). if None -- the same as fsfdir
    collapse_other_conditions: collapse all other conditions into a single regressor for
        the lsone model.  Jeanette's analyses suggest that it's better than leaving
        them separate.
    data_file: allows to override path of the 4D datafile instead of specified in design.fsf
        'feat_files(1)'
    """

    known_methods = ['lsall', 'lsone']
    assert set(methods).issubset(set(known_methods)), \
           "Unknown method(s): %s" % (set(methods).difference(set(known_methods)))

    if not os.path.exists(fsfdir):
        print 'ERROR: %s does not exist!' % fsfdir
        #return

    if not fsfdir.endswith('/'):
        fsfdir = ''.join([fsfdir, '/'])
    if modeldir is None:
        modeldir = fsfdir

    # load design using pymvpa tools

    fsffile = os.path.join(fsfdir, design_fsf_file)
    desmatfile = os.path.join(modeldir, design_mat_file)

    verbose(1, "Loading design")
    design = read_fsl_design(fsffile)

    desmat = FslGLMDesign(desmatfile)

    ntp, nevs = desmat.mat.shape

    TR = design['fmri(tr)']

    hrf = spm_hrf(time_res)


    # Set up the high time-resolution design matrix
    
    time_up = N.arange(0, TR*ntp+time_res, time_res)
    n_up = len(time_up)

    # exclude events that occur within two TRs of the end of the run, due to the
    # inability to accurately estimate the response to them.
    
    max_evtime = TR*ntp - 2;
    
    good_evs = []
    motion_evs = []
    ons = []
    
    if outdir is None:
        outdir = os.path.join(fsfdir, 'betaseries')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # load data
    verbose(1, "Loading data")

    maskimg = os.path.join(fsfdir, mask_file or 'mask.nii.gz')
    data = fmri_dataset(os.path.join(fsfdir, data_file or design['feat_files']),
                        mask=maskimg)
    # mask = fmri_dataset(maskimg, mask=maskimg)
    nvox = data.nfeatures


    # create smoothing kernel for design
    cutoff = design['fmri(paradigm_hp)']/TR
    verbose(1, "Creating smoothing kernel based on the original analysis cutoff %.2f"
               % cutoff)
    F = get_smoothing_kernel(cutoff, ntp)

    verbose(1, "Determining non-motion conditions")
    # loop through and find the good (non-motion) conditions
    # NB: this assumes that the name of the motion EV includes "motpar"
    # ala the openfmri convention.
    # TO DO:  add ability to manually specify motion regressors (currently assumes
    # that any EV that includes "motpar" in its name is a motion regressor)
    evctr = 0
    ev_td = N.zeros(design['fmri(evs_real)'])

    for ev in range(design['fmri(evs_orig)']):
        # filter out motion parameters
        evtitle = design['fmri(evtitle%d)'%int(ev+1)]
        verbose(2, "Loading %s" % evtitle)
        if evtitle.find('mot')!=0:
            good_evs.append(evctr)
            evctr+=1
            if design['fmri(deriv_yn%d)'%int(ev+1)]==1:
                ev_td[evctr] = 1
                # skip temporal derivative
                evctr+=1
            ons.append(FslEV3(os.path.join(fsfdir, design['fmri(custom%d)'%int(ev+1)])))
        else:
            motion_evs.append(evctr)
            evctr+=1
            if design['fmri(deriv_yn%d)'%int(ev+1)]==1:
                # skip temporal derivative
                ev_td[evctr] = 1
                motion_evs.append(evctr)
                evctr+=1

    ntrials_total = 0
    
    for x in range(len(good_evs)):
        ntrials_total = ntrials_total+len(ons[x]['onsets'])

    dm_nuisanceevs = desmat.mat[:, motion_evs]

    if 'lsone' in methods:
        # loop through the good evs and build the ls-one model
        # design matrix for each trial/ev
        method = 'lsone'
        verbose(1, 'Estimating ls-one model...')
        trial_ctr = 0
        all_conds = []
        beta_maker = N.zeros((ntrials_total, ntp))
        for e in range(len(good_evs)):
            ev = good_evs[e]
            # first, take the original desmtx and remove the ev of interest
            other_good_evs = [x for x in good_evs if x != ev]
            # put the temporal derivatives in
            og = copy(other_good_evs)
            for x in og:
                if ev_td[x] > 0:
                    other_good_evs.append(x+1)
            dm_otherevs = desmat.mat[:, other_good_evs]
            cond_ons = N.array(ons[e].onsets)
            cond_dur = N.array(ons[e].durations)
            ntrials = len(cond_ons)
            glm_res_full = N.zeros((nvox, ntrials))
            verbose(2, 'processing ev %d: %d trials' % (e+1, ntrials))
            for t in range(ntrials):
                all_conds.append((ev/2)+1)
                if cond_ons[t] > max_evtime:
                    print 'TOI: skipping ev %d trial %d: %f %f'%(ev, t, cond_ons[t], max_evtime)
                    trial_ctr+=1
                    continue
                # first build model for the trial of interest at high resolution
                dm_toi = N.zeros(n_up)
                window_ons = [N.where(time_up==x)[0][0] for x in time_up if (x > cond_ons[t]) & (x < cond_ons[t] + cond_dur[t])]
                dm_toi[window_ons] = 1
                dm_toi = N.convolve(dm_toi, hrf)[0:ntp/time_res*TR:(TR/time_res)]
                other_trial_ons = cond_ons[N.where(cond_ons!=cond_ons[t])[0]]
                other_trial_dur = cond_dur[N.where(cond_ons!=cond_ons[t])[0]]
                
                dm_other = N.zeros(n_up)
                # process the other trials
                for o in other_trial_ons:
                    if o > max_evtime:
                        continue
                    # find the timepoints that fall within the window b/w onset and onset + duration
                    window_ons = [N.where(time_up==x)[0][0] for x in time_up if (x > o) & (x < o + other_trial_dur[N.where(other_trial_ons==o)[0][0]])]
                    dm_other[window_ons] = 1
                    
                # Put together the design matrix
                dm_other = N.convolve(dm_other, hrf)[0:ntp/time_res*TR:(TR/time_res)]
                if collapse_other_conditions:
                    dm_other = N.hstack((N.dot(F, dm_other[0:ntp, N.newaxis]), dm_otherevs))
                    dm_other = N.sum(dm_other, 1)
                    dm_full = N.hstack((N.dot(F, dm_toi[0:ntp, N.newaxis]), dm_other[:, N.newaxis], dm_nuisanceevs))
                else:
                    dm_full = N.hstack((N.dot(F, dm_toi[0:ntp, N.newaxis]), N.dot(F, dm_other[0:ntp, N.newaxis]), dm_otherevs, dm_nuisanceevs))
                dm_full = dm_full - N.kron(N.ones((dm_full.shape[0], dm_full.shape[1])), N.mean(dm_full, 0))[0:dm_full.shape[0], 0:dm_full.shape[1]]
                dm_full = N.hstack((dm_full, N.ones((ntp, 1))))
                beta_maker_loop = N.linalg.pinv(dm_full)
                beta_maker[trial_ctr, :] = beta_maker_loop[0, :]
                trial_ctr+=1
        # this uses Jeanette's trick of extracting the beta-forming vector for each
        # trial and putting them together, which allows estimation for all trials
        # at once
        
        glm_res_full = N.dot(beta_maker, data.samples)
 
        # map the data into images and save to betaseries directory
        
        all_conds = N.array(all_conds)
        for e in range(len(good_evs)):
            ni = map2nifti(data, data=glm_res_full[N.where(all_conds==(e+1))[0], :])
            ni.to_filename(fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e+1), method))


    if 'lsall' in methods:  # do ls-all
       method = 'lsall'
       print 'estimating ls-all...'
       # first get all onsets in a row
       all_onsets = []
       all_durations = []
       all_conds = []  # condition marker
       for e in range(len(good_evs)):
            ev = good_evs[e]        
            all_onsets = N.hstack((all_onsets, ons[e].onsets))
            all_durations = N.hstack((all_durations, ons[e].durations))
            all_conds = N.hstack((all_conds, N.ones(len(ons[e].onsets))*((ev/2)+1)))

       #all_onsets=N.round(all_onsets/TR)  # round to nearest TR number
       ntrials = len(all_onsets)
       glm_res_full = N.zeros((nvox, ntrials))
       dm_trials = N.zeros((ntp, ntrials))
       dm_full = []
       for t in range(ntrials):
                if all_onsets[t] > max_evtime:
                    continue
                # build model for each trial
                dm_trial = N.zeros(n_up)
                window_ons = [N.where(time_up==x)[0][0] for x in time_up if (x > all_onsets[t]) & (x < all_onsets[t] + all_durations[t])]
                dm_trial[window_ons] = 1
                dm_trial = N.convolve(dm_trial, hrf)[0:ntp/time_res*TR:(TR/time_res)]
                dm_trials[:, t] = dm_trial

       # filter the desmtx, except for the nuisance part (which is already filtered)
       if len(motion_evs)>0:
           dm_full = N.hstack((N.dot(F, dm_trials), dm_nuisanceevs))
       else:
           dm_full = N.dot(F, dm_trials)

       dm_full = dm_full - N.kron(N.ones((dm_full.shape[0], dm_full.shape[1])), N.mean(dm_full, 0))[0:dm_full.shape[0], 0:dm_full.shape[1]]
       dm_full = N.hstack((dm_full, N.ones((ntp, 1))))
       glm_res_full = N.dot(N.linalg.pinv(dm_full), data.samples)
       glm_res_full = glm_res_full[0:ntrials, :]

       #for v in range(nvox):
                    #try:
        #                glm_result = estimate_OLS(dm_full, data.samples[:, v])
         #               glm_res_full[v, :] = glm_result[0:ntrials]
                    #except:
                    #    print 'problem with trial %d, cond %d'%(t, e)


       for e in range(len(good_evs)):
           ni = map2nifti(data, data=glm_res_full[N.where(all_conds==(e+1))[0], :])
           ni.to_filename(fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e+1), method))




def spm_hrf(TR, p=[6, 16, 1, 1, 6, 0, 32]):
    """ An implementation of spm_hrf.m from the SPM distribution

    Arguments:

    Required:
    TR: repetition time at which to generate the HRF (in seconds)

    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32

    """

    p = [float(x) for x in p]

    fMRI_T = 16.0

    TR = float(TR)
    dt  = TR/fMRI_T
    u   = N.arange(p[6]/dt + 1) - p[5]/dt
    hrf = scipy.stats.gamma.pdf(u, p[0]/p[2], scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u, p[1]/p[3], scale=1.0/(dt/p[3]))/p[4]
    good_pts = N.array(range(N.int(p[6]/TR)))*fMRI_T
    hrf = hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/N.sum(hrf);
    return hrf



def estimate_OLS(desmtx, data, demean=1, resid=0):
    """A utility function to compute ordinary least squares

    Arguments:

    Required:
    desmtx: design matrix
    data: the data

    Optional:

    demean: demean the data before estimation
    resid: return the residuals
    """
    if demean == 1:      # use if desmtx doesn't include a constant
        data = data-N.mean(data)
    glm = N.linalg.lstsq(desmtx, data)
    if resid==1:  # return residuals as well
        return glm[0], glm[1]
    else:
        return glm[0]

if __name__ == '__main__':
    # #'/usr/share/fsl-feeds/data/fmri.feat/',
    if True:
        pybetaseries('/home/yoh/proj/pymvpa/pymvpa/3rd/pybetaseries/run001_test_data.feat',
                     design_fsf_file='design_yoh.fsf')
    else:
        topdir = '/data/famface/nobackup_pipe+derivs+nipymc/famface_level1/firstlevel'
        modelfit_dir = os.path.join(topdir, 'modelfit/_subject_id_km00/_fwhm_4.0/')
        mask_file = os.path.join(topdir, 'preproc/_subject_id_km00/meanfuncmask/corr_06mar11km_WIP_fMRI_SSh_3mm_sense2_sl35_SENSE_13_1_dtype_mean_brain_mask.nii.gz')
        pybetaseries(
                 os.path.join(modelfit_dir, 'level1design'),
                 design_fsf_file='run0.fsf',
                 modeldir=os.path.join(modelfit_dir, 'modelgen/mapflow/_modelgen0'),
                 design_mat_file='run0.mat',
                 mask_file=mask_file,
                 #methods=['lsone'],
                 outdir='/tmp/betaseries')
