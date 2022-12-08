import numpy as np 

def cell_bboxes_to_bboxes(cell_bboxes):
    """
    cell_boxes -> array : shape(S,S,5+C)
    Takes S*S*(5+C) returns a list of bboxes of size S*S
    list:[[p,x,y,w,h,category], ....] 
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

def mAP():
    pass

def IOU(bboxes_true, bboxes_pred):
    """
    bbox (list) len N: [[x,y,w,h], ...] 
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
    box2_area = bboxes_pred[: ,3] * bboxes_pred[: ,3] 
    union_area = (box1_area + box2_area - intersection_area + 1e-6)
    return intersection_area/union_area