import numpy as np 

def cell_bboxes_to_bboxes(cell_bboxes):
    """
    cell_boxes: numpy.array of shape(S,S,5+C)
    returns: a list of bboxes of size S*S*6 shape(S*S ,6)
    [[p,x,y,w,h,category], ....] 
    
    """
    S = cell_bboxes.shape[0]
    bboxes = []
    for row in range(S):
        for col in range(S):
            box = cell_bboxes[row, col, 0:5+1]
            box[5] = np.argmax(cell_bboxes[row, col, 5:])
            bboxes.append(box)

    return bboxes

def keep_bboxes(bboxes, thresh_hold):
    """
    bboxes -> list: [[p,x,y,h,w,category], ...]
    keeps the bboxes that have p>thresh_hold
    """
    filtered_bboxes = []
    for box in bboxes:
        if box[0]>thresh_hold:
            filtered_bboxes.append(box)
    return filtered_bboxes


def plot_image_with_bboxes(image, cell_bboxes, thresh_hold =0.5):
    """
    image: numpy array of shape(height, width, 3)
    cell_boxes:numpy array of shape(S,S,5+C) S*S*[p,x,y,w,h, ...]
    """
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    category2name = {0:"Aeroplane", 1:"Bicycle", 2:"Bird", 3:"Boat", 4:"Bottle", 5:"Bus", 6:"Car", 7:"Cat", 8:"Chair", 9:"Cow", 10:"Diningtable", 11:"Dog", 12:"Horse", 13:"Motorbike", 14:"Person", 15:"Pottedplant", 16:"Sheep", 17:"Sofa", 18:"Train", 19:"Tvmonitor"}
    category2color = {0:"lightcoral", 1:"red", 2:"sandybrown", 3:"linen", 4:"darkorange", 5:"gold", 6:"yellow", 7:"greenyellow", 8:"forestgreen", 9:"aquamarine", 10:"cyan", 11:"blue", 12:"blueviolet", 13:"violet", 14:"orange", 15:"peru", 16:"purple", 17:"thistle", 18:"pink", 19:"crimson"}
    bboxes = cell_bboxes_to_bboxes(cell_bboxes)
    bboxes = keep_bboxes(bboxes, thresh_hold)

    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)

    img_height, img_width, _ = image.shape
    for bbox in bboxes:
        p, center_x, center_y, width, height, category = bbox
        x = center_x * img_width
        y = center_y * img_height
        w = width * img_width
        h = height * img_height 
        x = x - w/2
        y = y - h/2
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=category2color[int(category)], facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, category2name[int(category)], bbox=dict(facecolor=category2color[int(category)], alpha=1.0))


    ax.imshow(image)
    plt.show()

def mAP(cell_bboxes_true, cell_bboxes_pred, iou_thresh_hold = 0.5, detection_thresh_hold = 0.5, num_of_classes = 20):
    """
    cell_bboxes_true: numpy array of shape(N,S,S,5+C)
    N: number of images
    S*S: number of cells
    5+C: 5 for [p,x,y,w,h], C for the number of categories in one hot encoding
    the same for cells_bboxes_pred  
    """
    #bboxes_true, bboxes_pred: list of list [p,x,y,w,h,category]
    bboxes_true = []
    bboxes_pred = []
    for cell_bboxes in cell_bboxes_true:
        bboxes = cell_bboxes_to_bboxes(cell_bboxes)#returns list of len N, where N is the number of images, containing a list of boxes [[p,x,y,w,y,category], ....]
        bboxes = keep_bboxes(bboxes, detection_thresh_hold)#keeps bboxes with p>detection_thresh_hold
        bboxes_true.append(bboxes)
    
    for cell_bboxes in cell_bboxes_pred:
        bboxes = cell_bboxes_to_bboxes(cell_bboxes)#returns list of len N, where N is the number of images, containing a list of boxes [[p,x,y,w,y,category], ....]
        bboxes = keep_bboxes(bboxes, detection_thresh_hold)#keeps bboxes with p>detection_thresh_hold
        bboxes_pred.append(bboxes)
    
    TP = []
    FP = []
    DC = [] # detection confidence 
    for c in range(num_of_classes):
        TP.append([])
        FP.append([])
        DC.append([])
        for image_id in range(len(bboxes_pred)):
            bboxes_true_of_image = bboxes_true[image_id]
            bboxes_pred_of_image = bboxes_pred[image_id]

            #bboxes_true_of_image for specific category
            bboxes_true_of_image_for_cat = []
            bboxes_pred_of_image_for_cat = []

            for bbox in bboxes_true_of_image:
                if bbox[-1] == c:
                    bboxes_true_of_image_for_cat.append(bbox)
            for bbox in bboxes_pred_of_image:
                if bbox[-1] == c:
                    bboxes_pred_of_image_for_cat.append(bbox)
            
            tp = np.zeros(len(bboxes_pred_of_image_for_cat))
            fp = np.zeros(len(bboxes_pred_of_image_for_cat))+1
            dc = [bbox[0] for bbox in bboxes_pred_of_image_for_cat]#detection confidence

            ground_truths_best_detections = [[] for _ in bboxes_true_of_image_for_cat]# a list that containes a list of detections for each ground truth, each list containes the ious and ids of a predicted boxes for that ground thruth 

            for bbox_pred_id, bbox_pred in enumerate(bboxes_pred_of_image_for_cat):
                best_iou = -1 
                best_iou_id = -1
                for bbox_true_id, bbox_true in enumerate(bboxes_true_of_image_for_cat):
                    iou = IOU(np.array([bbox_true]), np.array([bbox_pred]))
                    if iou>best_iou:
                        best_iou = iou
                        best_iou_id = bbox_true_id
               
                if best_iou>iou_thresh_hold:   
                    ground_truths_best_detections[best_iou_id].append([best_iou, bbox_pred_id])#we append the detection to the ground truth that we think that it belongs, if it has a iou > iou_thresh_hold

            for ground_truth_best_detections in ground_truths_best_detections:
                best_detection_iou = -1
                best_detection_iou_id = -1
                for detection_id, detection in enumerate(ground_truth_best_detections):
                    iou, id = detection[0], detection_id
                    if iou > best_detection_iou:
                        best_detection_iou = iou
                        best_detection_iou_id = id 
                if best_detection_iou_id !=-1:
                    best_detection = ground_truth_best_detections[best_detection_iou_id]
                    tp[best_detection[1]] = 1
                    fp[best_detection[1]] = 0
 
            TP[c].append(tp)
            TP[c].append(fp)
            DC[c].append(dc)
    
    #to do 
    print(TP)
    print(FP)
    print(DC)
    # s = 0
    # for b in bboxes_true:
    #     for _ in b:
    #         s+=1
    # print(s)
                
                
def IOU(bboxes_true, bboxes_pred):
    """
    bbox numpy array shape(N,4): [[x,y,w,h], ...] 
    x,y is the center of the box 
    returns list of IOUs len N: [iou1, iou2, ...]
    """
    dx = -np.abs(bboxes_true[: ,0] - bboxes_pred[: ,0]) + bboxes_true[: ,2] * 0.5 + bboxes_pred[: ,2] * 0.5  
    dy = -np.abs(bboxes_true[: ,1] - bboxes_pred[: ,1]) + bboxes_true[: ,3] * 0.5 + bboxes_pred[: ,3] * 0.5
    dx = np.maximum(dx, 0)
    dy = np.maximum(dy, 0)

    mdx = np.minimum(bboxes_true[: ,2], bboxes_pred[: ,2])
    mdy = np.minimum(bboxes_true[: ,3], bboxes_pred[: ,3])

    hx = np.minimum(mdx, dx)
    hy = np.minimum(mdy, dy)

    intersection_area = hx * hy
    box1_area = bboxes_true[: ,2] * bboxes_true[: ,3] 
    box2_area = bboxes_pred[: ,2] * bboxes_pred[: ,3] 
    union_area = (box1_area + box2_area - intersection_area + 1e-6)
    return intersection_area/union_area