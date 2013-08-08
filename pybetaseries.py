#!/usr/bin/env python
"""pybetaseries: a module for computing beta-series regression on fMRI data

Includes:
pybetaseries: main function
estimate_OLS: helper function to estimate least squares model
spm_hrf: helper function to generate double-gamma HRF
"""

from glob import glob

from mvpa2.base import verbose
from mvpa2.misc.fsl.base import *
from mvpa2.datasets.mri import fmri_dataset, map2nifti

import numpy as N
import nibabel
import scipy.stats
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
#from mvpa.datasets.mri import *
import os
from os.path import join as pjoin
from copy import copy


def complete_filename(fileprefix):
    """Check if provided fileprefix is not pointing to an existing file but there exist a single file with some .extension for it"""
    if os.path.exists(fileprefix):
        return fileprefix
    filenames = glob(fileprefix + ".*")
    if len(filenames) > 1:
        # bloody pairs
        if sorted(filenames) == [fileprefix + '.hdr', fileprefix + '.img']:
            return fileprefix + '.hdr'
        raise ValueError("There are multiple files available with prefix %s: %s"
                         % (fileprefix, ", ".join(filenames)))
    elif len(filenames) == 1:
        return filenames[0]
    else:
        raise ValueError("There are no files for %s" % fileprefix)

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

# yoh:
#  desmat -- theoretically should be "computed", not loaded
#  time_up and hrf as sequences -- might better be generated "inside"
#      since they are not "independent".  Note that time_res is now
#      computed inside.  RFing in favor of passing TR, time_res
#  TR -- theoretically should be available in data
def extract_lsone(data, TR, time_res,
                  hrf_gen, F,
                  good_ons,
                  good_evs, nuisance_evs, withderiv_evs,
                  desmat,
                  extract_evs=None,
                  collapse_other_conditions=True):
    # loop through the good evs and build the ls-one model
    # design matrix for each trial/ev

    ntp, nvox = data.shape

    hrf = hrf_gen(time_res)
    # Set up the high time-resolution design matrix
    time_up = N.arange(0, TR*ntp+time_res, time_res)
    n_up = len(time_up)
    dm_nuisanceevs = desmat.mat[:, nuisance_evs]

    ntrials_total = sum(len(o['onsets']) for o in good_ons)
    verbose(1, "Have %d trials total to process" % ntrials_total)
    trial_ctr = 0
    all_conds = []
    beta_maker = N.zeros((ntrials_total, ntp))

    if extract_evs is None:
        extract_evs = range(len(good_evs))

    for e in extract_evs: # range(len(good_evs)):
        ev = good_evs[e]
        # first, take the original desmtx and remove the ev of interest
        other_good_evs = [x for x in good_evs if x != ev]
        # put the temporal derivatives into other_good_evs
        # start with its own derivative.  This accounts for
        # a significant amount of divergence from matlab implementation
        if ev in withderiv_evs:
            other_good_evs.append(ev+1)
        for x in other_good_evs:
            if x in withderiv_evs:
                other_good_evs.append(x+1)
        dm_otherevs = desmat.mat[:, other_good_evs]
        cond_ons = N.array(good_ons[e].onsets)
        cond_dur = N.array(good_ons[e].durations)
        ntrials = len(cond_ons)
        glm_res_full = N.zeros((nvox, ntrials))
        verbose(2, 'processing ev %d: %d trials' % (e+1, ntrials))
        for t in range(ntrials):
            verbose(3, "processing trial %d" % t)
            ## ad-hoc warning -- assumes interleaved presence of
            ## derivatives' EVs
            all_conds.append((ev/2)+1)
            ## yoh: handle outside
            ## if cond_ons[t] > max_evtime:
            ##     verbose(1, 'TOI: skipping ev %d trial %d: %f %f'
            ##                % (ev, t, cond_ons[t], max_evtime))
            ##     trial_ctr += 1
            ##     continue
            # first build model for the trial of interest at high resolution
            dm_toi = N.zeros(n_up)
            window_ons = [N.where(time_up==x)[0][0]
                          for x in time_up
                          if (x >= cond_ons[t]) & (x < cond_ons[t] + cond_dur[t])]
            dm_toi[window_ons] = 1
            dm_toi = N.convolve(dm_toi, hrf)[0:ntp/time_res*TR:(TR/time_res)]
            other_trial_ons = cond_ons[N.where(cond_ons!=cond_ons[t])[0]]
            other_trial_dur = cond_dur[N.where(cond_ons!=cond_ons[t])[0]]

            dm_other = N.zeros(n_up)
            # process the other trials
            for o in other_trial_ons:
                ## yoh: handle outside
                ## if o > max_evtime:
                ##     continue
                # find the timepoints that fall within the window b/w onset and onset + duration
                window_ons = [N.where(time_up==x)[0][0]
                              for x in time_up
                              if o <= x < o + other_trial_dur[N.where(other_trial_ons==o)[0][0]]]
                dm_other[window_ons] = 1

            # Put together the design matrix
            dm_other = N.convolve(dm_other, hrf)[0:ntp/time_res*TR:(TR/time_res)]
            if collapse_other_conditions:
                dm_other = N.hstack((N.dot(F, dm_other[0:ntp, N.newaxis]), dm_otherevs))
                dm_other = N.sum(dm_other, 1)
                dm_full = N.hstack((N.dot(F, dm_toi[0:ntp, N.newaxis]),
                                    dm_other[:, N.newaxis], dm_nuisanceevs))
            else:
                dm_full = N.hstack((N.dot(F, dm_toi[0:ntp, N.newaxis]),
                                    N.dot(F, dm_other[0:ntp, N.newaxis]),
                                    dm_otherevs,
                                    dm_nuisanceevs))
            dm_full -= dm_full.mean(0)
            dm_full = N.hstack((dm_full, N.ones((ntp, 1))))
            beta_maker_loop = N.linalg.pinv(dm_full)
            beta_maker[trial_ctr, :] = beta_maker_loop[0, :]
            trial_ctr += 1
    # this uses Jeanette's trick of extracting the beta-forming vector for each
    # trial and putting them together, which allows estimation for all trials
    # at once

    glm_res_full = N.dot(beta_maker, data.samples)

    return all_conds, glm_res_full


def extract_lsall(data, TR, time_res,
                  hrf_gen, F,
                  good_ons,
                  good_evs,
                  desmat,
                  extract_evs=None):
    ntp, nvox = data.shape

    hrf = hrf_gen(time_res)
    # Set up the high time-resolution design matrix
    time_up = N.arange(0, TR*ntp+time_res, time_res)

    all_onsets = []
    all_durations = []
    all_conds = []  # condition marker

    if extract_evs is None:
        extract_evs = range(len(good_evs))

    nuisance_evs = sorted(list(set(range(desmat.mat.shape[1])).difference(
        [good_evs[e] for e in extract_evs])))

    for e in extract_evs:
        ev = good_evs[e]
        all_onsets    = N.hstack((all_onsets,    good_ons[e].onsets))
        all_durations = N.hstack((all_durations, good_ons[e].durations))
        # yoh: ad-hoc warning -- it is marking with (ev/2)+1 (I guess)
        # assuming presence of derivatives EVs
        all_conds     = N.hstack((all_conds,     N.ones(len(good_ons[e].onsets))*((ev/2)+1)))

    #all_onsets=N.round(all_onsets/TR)  # round to nearest TR number
    ntrials = len(all_onsets)
    glm_res_full = N.zeros((nvox, ntrials))
    dm_trials = N.zeros((ntp, ntrials))
    dm_full = []
    for t in range(ntrials):
        verbose(2, "Estimating for trial %d" % t)

        ## yoh: TODO -- filter outside
        ## if all_onsets[t] > max_evtime:
        ##     continue
        # build model for each trial
        dm_trial = N.zeros(len(time_up))
        window_ons = [N.where(time_up==x)[0][0]
                      for x in time_up
                      if all_onsets[t] <= x < all_onsets[t] + all_durations[t]]
        dm_trial[window_ons] = 1
        dm_trial_up = N.convolve(dm_trial, hrf)
        dm_trial_down = dm_trial_up[0:ntp/time_res*TR:(TR/time_res)]
        dm_trials[:, t] = dm_trial_down

    # filter the desmtx, except for the nuisance part (which is already filtered)
    # since it is taken from a loaded FSL
    dm_full = N.dot(F, dm_trials)

    # mean center trials models
    dm_trials -= dm_trials.mean(0)

    if len(nuisance_evs) > 0:
        # and stick nuisance evs if any to the back
        dm_full = N.hstack((dm_full, desmat.mat[:, nuisance_evs]))

    dm_full = N.hstack((dm_full, N.ones((ntp, 1))))
    glm_res_full = N.dot(N.linalg.pinv(dm_full), data.samples)
    glm_res_full = glm_res_full[:ntrials]

    return all_conds, glm_res_full


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
                 extract_evs=None,
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

    fsffile = pjoin(fsfdir, design_fsf_file)
    desmatfile = pjoin(modeldir, design_mat_file)

    verbose(1, "Loading design")
    design = read_fsl_design(fsffile)

    desmat = FslGLMDesign(desmatfile)

    ntp, nevs = desmat.mat.shape

    TR = design['fmri(tr)']
    # yoh: theoretically it should be identical to the one read from
    # the nifti file, but in this sample data those manage to differ:
    # bold_mcf_brain.nii.gz        int16  [ 64,  64,  30, 182] 3.12x3.12x5.00x1.00   sform
    # filtered_func_data.nii.gz   float32 [ 64,  64,  30, 182] 3.12x3.12x5.00x2.00   sform
    #assert(abs(data.a.imghdr.get_zooms()[-1] - TR) < 0.001)
    # it is the filtered_func_data.nii.gz  which was used for analysis,
    # and it differs from bold_mcf_brain.nii.gz ... 

    # exclude events that occur within two TRs of the end of the run, due to the
    # inability to accurately estimate the response to them.

    max_evtime = TR*ntp - 2;
    # TODO: filter out here the trials jumping outside

    good_evs = []
    nuisance_evs = []
    # yoh: ev_td marks temporal derivatives (of good EVs or of nuisance -- all)
    #      replacing with deriv_evs for consistency
    withderiv_evs = []
    # ev_td = N.zeros(design['fmri(evs_real)'])

    good_ons = []

    if outdir is None:
        outdir = pjoin(fsfdir, 'betaseries')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # create smoothing kernel for design
    cutoff = design['fmri(paradigm_hp)']/TR
    verbose(1, "Creating smoothing kernel based on the original analysis cutoff %.2f"
               % cutoff)
    # yoh: Verify that the kernel is correct since it looks
    # quite ...
    F = get_smoothing_kernel(cutoff, ntp)

    verbose(1, "Determining non-motion conditions")
    # loop through and find the good (non-motion) conditions
    # NB: this assumes that the name of the motion EV includes "motpar"
    # ala the openfmri convention.
    # TO DO:  add ability to manually specify motion regressors (currently assumes
    # that any EV that includes "motpar" in its name is a motion regressor)
    evctr = 0

    for ev in range(1, design['fmri(evs_orig)']+1):
        # filter out motion parameters
        evtitle = design['fmri(evtitle%d)' % ev]
        verbose(2, "Loading EV %s" % evtitle)
        if not evtitle.startswith('mot'):
            good_evs.append(evctr)
            evctr += 1
            if design['fmri(deriv_yn%d)' % ev] == 1:
                withderiv_evs.append(evctr-1)
                # skip temporal derivative
                evctr += 1
            ev_events = FslEV3(pjoin(fsfdir, design['fmri(custom%d)' % ev]))
            good_ons.append(ev_events)
        else:
            nuisance_evs.append(evctr)
            evctr += 1
            if design['fmri(deriv_yn%d)' % ev] == 1:
                # skip temporal derivative
                withderiv_evs.append(evctr)
                nuisance_evs.append(evctr)
                evctr += 1

    # load data
    verbose(1, "Loading data")

    maskimg = pjoin(fsfdir, mask_file or 'mask.nii.gz')
    # yoh: TODO design['feat_files'] is not the one "of interest" since it is
    # the input file, while we would like to operate on pre-processed version
    # which is usually stored as filtered_func_data.nii.gz
    data_file_fullname = complete_filename(
        pjoin(fsfdir, data_file or "filtered_func_data.nii.gz"))
    data = fmri_dataset(data_file_fullname, mask=maskimg)
    assert(len(data) == ntp)

    for method in methods:
        verbose(1, 'Estimating %(method)s model...' % locals())

        if method == 'lsone':
            all_conds, glm_res_full = extract_lsone(
                        data, TR, time_res,
                        spm_hrf, F,
                        good_ons,
                        good_evs, nuisance_evs, withderiv_evs,
                        desmat,
                        extract_evs=extract_evs,
                        collapse_other_conditions=collapse_other_conditions)
        elif method == 'lsall':
            all_conds, glm_res_full = extract_lsall(
                        data, TR, time_res,
                        spm_hrf, F,
                        good_ons,
                        good_evs,
                        desmat,
                        extract_evs=extract_evs,
                        )
        else:
            raise ValueError(method)

        all_conds = N.asanyarray(all_conds)   # assure array here
        # map the data into images and save to betaseries directory
        for e in range(1, len(good_evs)+1):
            ni = map2nifti(data, data=glm_res_full[N.where(all_conds==e)[0], :])
            ni.to_filename(pjoin(outdir, 'ev%d_%s.nii.gz' % (e, method)))



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
    Gpdf = scipy.stats.gamma.pdf 
    hrf = Gpdf(u, p[0]/p[2], scale=1.0/(dt/p[2])) - Gpdf(u, p[1]/p[3], scale=1.0/(dt/p[3]))/p[4]
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

    verbose.level = 3
    # #'/usr/share/fsl-feeds/data/fmri.feat/',
    if True:
        topdir = '/home/yoh/proj/pymvpa/pymvpa/3rd/pybetaseries/run001_test_data.feat'
        pybetaseries(topdir,
                     time_res=2./16,       # just to make matlab code
                     methods=['lsone'],
                     design_fsf_file='design_yoh.fsf',
                     #mask_file='mask_small.hdr',
                     extract_evs=[2],
                     collapse_other_conditions=False,
                     outdir=pjoin(topdir, 'betaseries-yarikcode-3-nocollapse4'))
    else:
        topdir = '/data/famface/nobackup_pipe+derivs+nipymc/famface_level1/firstlevel'
        modelfit_dir = pjoin(topdir, 'modelfit/_subject_id_km00/_fwhm_4.0/')
        mask_file = pjoin(topdir, 'preproc/_subject_id_km00/meanfuncmask/corr_06mar11km_WIP_fMRI_SSh_3mm_sense2_sl35_SENSE_13_1_dtype_mean_brain_mask.nii.gz')
        pybetaseries(
                 pjoin(modelfit_dir, 'level1design'),
                 design_fsf_file='run0.fsf',
                 modeldir=pjoin(modelfit_dir, 'modelgen/mapflow/_modelgen0'),
                 design_mat_file='run0.mat',
                 mask_file=mask_file,
                 #methods=['lsone'],
                 outdir='/tmp/betaseries')
