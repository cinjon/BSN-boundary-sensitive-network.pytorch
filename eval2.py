import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')
    
    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])
        
        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) + (
            target_segments[i, 1] - target_segments[i, 0] + 1) - intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou
    
    
def average_recall_vs_nr_proposals(proposals,
                                   ground_truth,
                                   tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    """ Computes the average recall given an average number 
        of proposals per video.
        
    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include 
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include 
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    tiou_thresholds : 1darray, optional
        array with tiou threholds.
        
    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = proposals['video-name'].unique()
    
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]
        
        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        
        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)
        
    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.
    
    # Computes average recall.
    pcn_lst = np.arange(1, 201) / 200.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            
            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) >
                                 0).sum()
                
        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()
        
    # Recall is averaged.
    avg_recall = recall.mean(axis=0)
    
    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (
        float(proposals.shape[0]) / video_lst.shape[0])
    
    return recall, avg_recall, proposals_per_video
                
                
def recall_vs_tiou_thresholds(proposals,
                              ground_truth,
                              nr_proposals=1000,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    """ Computes recall at different tiou thresholds given a fixed 
        average number of proposals per video.
    
    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include 
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include 
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    nr_proposals : int
        average number of proposals per video.
    tiou_thresholds : 1darray, optional
        array with tiou threholds.
        
    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = proposals['video-name'].unique()
    
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        
        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]
        
        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values
        
        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)
        
    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    pcn = (video_lst.shape[0] * float(nr_proposals)) / proposals.shape[0]
        
    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            
            # Get number of proposals at the fixed percentage of total retrieved.
            nr_proposals = int(score.shape[1] * pcn)
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) >
                                0).sum()
            
        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()
            
    return recall, tiou_thresholds
            

def plot_metric(opt,
                average_nr_proposals,
                average_recall,
                recall,
                tiou_thresholds=np.linspace(0.5, 1.0, 11)):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    colors = [
        'k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo'
    ]
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals,
                recall[2 * idx, :],
                color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "], area=" +
                str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4,
                linestyle='--',
                marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(
        average_nr_proposals,
        average_recall,
        color=colors[0],
        label="tiou = 0.5:0.05:1.0," + " area=" +
        str(int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
        linewidth=4,
        linestyle='-',
        marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1],
              loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #plt.show()
    save_path = os.path.join(opt['postprocessed_results_dir'], 'evaluation_result.jpg')
    plt.savefig(save_path)

    
def evaluation_proposal(opt):
    if 'thumos' in opt['dataset']:
        bsn_results = pd.read_csv(os.path.join(opt['postprocessed_results_dir'], 'thumos14_results.csv'))
    elif 'gymnastics' in opt['dataset']:
        bsn_results = pd.read_csv(os.path.join(opt['postprocessed_results_dir'], 'gym_results.csv'))        
    ground_truth = pd.read_csv(opt['video_info'])
    
    # Computes average recall vs average number of proposals.
    recall, average_recall, average_nr_proposals = average_recall_vs_nr_proposals(
        bsn_results, ground_truth)
    area_under_curve = np.trapz(average_recall, average_nr_proposals)
    f = interp1d(average_nr_proposals, average_recall, axis=0)
    interp_results = [(k, f(k)) for k in [50, 100, 200, 500, 1000]]
    interp_str = ', '.join(['%d: %.4f' % (k, v) for k, v in interp_results])

    with open(os.path.join(opt['postprocessed_results_dir'], 'output.txt'), 'w') as f:
        f.write('[RESULTS] Performance on %s proposal task.\n' % opt['dataset'])
        f.write('\tArea Under the AR vs AN curve: {}%\n'.format(
            100. * float(area_under_curve) / average_nr_proposals[-1]))
        f.write('Interpolation results: %s\n' % interp_str)
    
    plot_metric(opt, average_nr_proposals, average_recall, recall)

    
    
