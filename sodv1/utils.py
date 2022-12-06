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
