import numpy as np
import pandas as pd
import os


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def NMS(df, nms_threshold=0.75):
    df = df.sort(columns="score", ascending=False)
    
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    
    rstart = []
    rend = []
    rscore = []
    
    while len(tstart) > 1:
        idx = 1
        while idx < len(tstart):
            if IOU(tstart[0], tend[0], tstart[idx], tend[idx]) > nms_threshold:
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                idx += 1
                rstart.append(tstart[0])
                rend.append(tend[0])
                rscore.append(tscore[0])
                tstart.pop(0)
                tend.pop(0)
                tscore.pop(0)
    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf
            
            
def Soft_NMS(df, nms_threshold=0.75):
    df = df.sort_values(by="score", ascending=False)
    
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    
    rstart = []
    rend = []
    rscore = []
    
    while len(tscore) > 1 and len(rscore) < 1500:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx],
                              tend[idx])
                tmp_width = tend[max_index] - tstart[max_index]
                tmp_width = tmp_width / 300
                if tmp_iou > 0.5 + 0.3 * tmp_width:  #*1/(1+np.exp(-max_index)):
                    tscore[idx] = tscore[idx] * np.exp(
                        -np.square(tmp_iou) / 0.75)
                    
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        
    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    
    return newDf


def min_max(x):
    x = (x - min(x)) / (max(x) - min(x))
    return x


def BSN_post_processing(opt):
    annoDf = pd.read_csv(opt['video_info'])
    # "./data/thumos14_annotations/thumos14_test_groundtruth.csv")
    videoNameList = sorted(list(set(annoDf["video-name"].values[:])))
    # random.shuffle(videoNameList)

    xmin_list = []
    xmax_list = []
    score_list = []
    frame_list = []
    video_list = []
    
    pem_inference_results = opt['pem_inference_results_dir']
    for num, video_name in enumerate(videoNameList):
        if num % 25 == 0:
            print(num, len(videoNameList), name)
        videoAnno = annoDf[annoDf["video-name"] == video_name]
        videoFrame = videoAnno["video-frames"].values[0]
        try:
            df = pd.read_csv(os.path.join(pem_inference_results, video_name + ".csv"))
        except Exception as e:
            print("Nothing for this video ... %s" % video_name)
            continue

        df['score'] = df.iou_score.values[:] * df.xmin_score.values[:] * df.xmax_score.values[:]
        sdf = Soft_NMS(df, 0.5)
        for j in range(min(1500, len(sdf))):
            xmin_list.append(sdf.xmin.values[j])
            xmax_list.append(sdf.xmax.values[j])
            score_list.append(sdf.score.values[j])
            frame_list.append(videoFrame)
            video_list.append(video_name)
        
    outDf = pd.DataFrame()
    outDf["f-end"] = xmax_list
    outDf["f-init"] = xmin_list
    outDf["score"] = score_list
    outDf["video-frames"] = frame_list
    outDf["video-name"] = video_list

    output_dir = opt['postprocessed_results_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outfile = os.path.join(output_dir, 'thumos14_results.csv')
    outDf.to_csv(outfile, index=False)
