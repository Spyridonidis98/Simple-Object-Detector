import numpy as np 

def cell_bboxes_to_bboxes(cell_bboxes):
    S = cell_bboxes.shape[0]
    bboxes = []
    for row in range(S):
        for col in range(S):
            bboxes.append(cell_bboxes[row, col, 0:5])

    thresh_hold =0.5
    filtered_bbboxes = []
    for bbox in bboxes:
        if bbox[0]>thresh_hold:
            filtered_bbboxes.append(bbox)
    return filtered_bbboxes

def plot_image_with_bboxes(image, cell_bboxes):
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    
    bboxes = cell_bboxes_to_bboxes(cell_bboxes)

    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)

    img_height, img_width, _ = image.shape
    for bbox in bboxes:
        category, center_x, center_y, width, height = bbox
        x = center_x * img_width
        y = center_y * img_height
        w = width * img_width
        h = height * img_height 
        x = x - w/2
        y = y - h/2

        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.imshow(image)
    plt.show()

def mAP():
    pass

def IOU(bboxes_true, bboxes_pred):
    """
    bbox (list) : [x,y,w,h] 
    x,y is the center of the box 
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