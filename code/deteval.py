import math
from collections import namedtuple
from copy import deepcopy

import numpy as np


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT' : 0.8,
        'AREA_PRECISION_CONSTRAINT' : 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O':1.,
        'MTYPE_OM_O':0.8,
        'MTYPE_OM_M':1.,
        'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
        'CRLF':False # Lines are delimited by Windows CRLF format
    }


def calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict=None,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    현재는 rect(xmin, ymin, xmax, ymax) 형식의 bounding box만 지원함. 다른 형식(quadrilateral,
    poligon, etc.)의 데이터가 들어오면 외접하는 rect로 변환해서 이용하고 있음.
    """

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row,j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,j] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[i,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                cont = cont +1
        if (cont != 1):
            return False

        if recallMat[row,col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row,col] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True
        return False

    def num_overlaps_gt(gtNum):
        cont = 0
        for detNum in range(len(detRects)):
            if detNum not in detDontCareRectsNum:
                if recallMat[gtNum,detNum] > 0 :
                    cont = cont +1
        return cont

    def num_overlaps_det(detNum):
        cont = 0
        for gtNum in range(len(recallMat)):
            if gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum,detNum] > 0 :
                    cont = cont +1
        return cont

    def is_single_overlap(row, col):
        if num_overlaps_gt(row)==1 and num_overlaps_det(col)==1:
            return True
        else:
            return False

    def one_to_many_match(gtNum):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                if precisionMat[gtNum,detNum] >= eval_hparams['AREA_PRECISION_CONSTRAINT'] :
                    many_sum += recallMat[gtNum,detNum]
                    detRects.append(detNum)
        if round(many_sum,4) >=eval_hparams['AREA_RECALL_CONSTRAINT'] :
            return True,detRects
        else:
            return False,[]

    def many_to_one_match(detNum):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum,detNum] >= eval_hparams['AREA_RECALL_CONSTRAINT'] :
                    many_sum += precisionMat[gtNum,detNum]
                    gtRects.append(gtNum)
        if round(many_sum,4) >=eval_hparams['AREA_PRECISION_CONSTRAINT'] :
            return True,gtRects
        else:
            return False,[]

    def area(a, b):
            dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
            dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
            if (dx>=0) and (dy>=0):
                    return dx*dy
            else:
                    return 0.

    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x,y)

    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty )


    def center_distance(r1, r2):
        return point_distance(center(r1), center(r2))

    def diag(r):
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    if bbox_format != 'rect':
        raise NotImplementedError

    # bbox들이 rect 이외의 형식으로 되어있는 경우 rect 형식으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict= deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)
    for sample_name, bboxes in _gt_bboxes_dict.items():
        # 원래 rect 형식이었으면 변환 없이 그대로 이용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    perSampleMetrics = {}

    methodRecallSum = 0
    methodPrecisionSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    for sample_name in gt_bboxes_dict:

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCareRectsNum = []  # Array of Ground Truth Rectangles' keys marked as don't Care
        detDontCareRectsNum = []  # Array of Detected Rectangles' matched with a don't Care GT
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        pointsList = gt_bboxes_dict[sample_name]

        if transcriptions_dict is None:
            transcriptionsList = None
        else:
            transcriptionsList = transcriptions_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())
            if dontCare:
                gtDontCareRectsNum.append( len(gtRects)-1 )

        evaluationLog += "GT rectangles: " + str(len(gtRects)) + (" (" + str(len(gtDontCareRectsNum)) + " don't care)\n" if len(gtDontCareRectsNum)>0 else "\n")

        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())
                if len(gtDontCareRectsNum)>0 :
                    for dontCareRectNum in gtDontCareRectsNum:
                        dontCareRect = gtRects[dontCareRectNum]
                        intersected_area = area(dontCareRect,detRect)
                        rdDimensions = ( (detRect.xmax - detRect.xmin+1) * (detRect.ymax - detRect.ymin+1))
                        if (rdDimensions==0) :
                            precision = 0
                        else:
                            precision= intersected_area / rdDimensions
                        if (precision > eval_hparams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCareRectsNum.append( len(detRects)-1 )
                            break

            evaluationLog += "DET rectangles: " + str(len(detRects)) + (" (" + str(len(detDontCareRectsNum)) + " don't care)\n" if len(detDontCareRectsNum)>0 else "\n")

            if len(gtRects)==0:
                recall = 1
                precision = 0 if len(detRects)>0 else 1

            if len(detRects)>0:
                #Calculate recall and precision matrixs
                outputShape=[len(gtRects),len(detRects)]
                recallMat = np.empty(outputShape)
                precisionMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtRects),np.int8)
                detRectMat = np.zeros(len(detRects),np.int8)
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        intersected_area = area(rG,rD)
                        rgDimensions = ( (rG.xmax - rG.xmin+1) * (rG.ymax - rG.ymin+1) )
                        rdDimensions = ( (rD.xmax - rD.xmin+1) * (rD.ymax - rD.ymin+1))
                        recallMat[gtNum,detNum] = 0 if rgDimensions==0 else  intersected_area / rgDimensions
                        precisionMat[gtNum,detNum] = 0 if rdDimensions==0 else intersected_area / rdDimensions

                # Find one-to-one matches
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum :
                            match = one_to_one_match(gtNum, detNum)
                            if match is True :
                                #in deteval we have to make other validation before mark as one-to-one
                                if is_single_overlap(gtNum, detNum) is True :
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD)
                                    normDist /= diag(rG) + diag(rD)
                                    normDist *= 2.0
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR'] :
                                        gtRectMat[gtNum] = 1
                                        detRectMat[detNum] = 1
                                        recallAccum += eval_hparams['MTYPE_OO_O']
                                        precisionAccum += eval_hparams['MTYPE_OO_O']
                                        pairs.append({'gt':gtNum,'det':detNum,'type':'OO'})
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"
                # Find one-to-many matches
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    if gtNum not in gtDontCareRectsNum:
                        match,matchesDet = one_to_many_match(gtNum)
                        if match is True :
                            evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                            #in deteval we have to make other validation before mark as one-to-one
                            if num_overlaps_gt(gtNum)>=2 :
                                gtRectMat[gtNum] = 1
                                recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O'])
                                precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)==1 else eval_hparams['MTYPE_OM_O']*len(matchesDet))
                                pairs.append({'gt':gtNum,'det':matchesDet,'type': 'OO' if len(matchesDet)==1 else 'OM'})
                                for detNum in matchesDet :
                                    detRectMat[detNum] = 1
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                            else:
                                evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # Find many-to-one matches
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    if detNum not in detDontCareRectsNum:
                        match,matchesGt = many_to_one_match(detNum)
                        if match is True :
                            #in deteval we have to make other validation before mark as one-to-one
                            if num_overlaps_det(detNum)>=2 :
                                detRectMat[detNum] = 1
                                recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M']*len(matchesGt))
                                precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt)==1 else eval_hparams['MTYPE_OM_M'])
                                pairs.append({'gt':matchesGt,'det':detNum,'type': 'OO' if len(matchesGt)==1 else 'MO'})
                                for gtNum in matchesGt :
                                    gtRectMat[gtNum] = 1
                                evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                            else:
                                evaluationLog += "Match Discarded GT #" + str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"

                numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects)>0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision =  float(0) if (len(detRects) - len(detDontCareRectsNum))==0 else float(precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
                hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)

        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects) - len(gtDontCareRectsNum)
        numDet += len(detRects) - len(detDontCareRectsNum)

        perSampleMetrics[sample_name] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recall_matrix': [] if len(detRects)>100 else recallMat.tolist(),
            'precision_matrix': [] if len(detRects)>100 else precisionMat.tolist(),
            'gt_bboxes': gtPolPoints,
            'det_bboxes': detPolPoints,
            'gt_dont_care': gtDontCareRectsNum,
            'det_dont_care': detDontCareRectsNum,
        }

        if verbose:
            perSampleMetrics[sample_name].update(evaluation_log=evaluationLog)

    methodRecall = 0 if numGt==0 else methodRecallSum/numGt
    methodPrecision = 0 if numDet==0 else methodPrecisionSum/numDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall,'hmean': methodHmean}

    resDict = {'calculated': True, 'Message': '', 'total': methodMetrics,
               'per_sample': perSampleMetrics, 'eval_hparams': eval_hparams}

    return resDict
