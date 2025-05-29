## Measurement code module

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def find_rectangular_width(mask, center=None):
    """
    Finds the width of the rectangular object in a 512x512 binary mask
    in a way that best hugs its rotated edges. If a center (y, x) is given,
    only contours with center-of-mass above that y value are considered.
    If center is not provided, returns the average width among all contours.

    Parameters:
      mask (2D numpy array): A binary 512x512 mask.
      center (tuple, optional): (y, x) coordinate. Only consider contours whose
                                center-of-mass y is less than center[0].

    Returns:
      - If center is provided:
            best_box (numpy array): The 4 vertices of the best–fitting rotated
                                    rectangle (from cv2.boxPoints). The two edges
                                    corresponding to the smaller side have been
                                    identified as the object's width boundaries.
      - Otherwise:
            avg_width (float): The average width among all contours.
    """
    # Find external contours; ensure mask is uint8.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []    # rotated rectangle vertices for each contour
    widths = []   # the smaller side (width) of each rectangle
    reported_width = 69
    for cnt in contours:
        if cv2.contourArea(cnt) == 0:
            continue

        # Compute center of mass using moments.
        M = cv2.moments(cnt)
        cY = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
        
        if center is not None and cY >= center[0]:
            continue

        # Get the minimum area rectangle that encloses the contour.
        rect = cv2.minAreaRect(cnt)  # rect = ((cx,cy), (w,h), angle)
        box = cv2.boxPoints(rect)      # get 4 vertices
        box = np.intp(box)
        reported_width = rect[1][0]
        # By convention, choose the smaller side as the true width.
        w, h = rect[1]
        rect_width = min(w, h)
        widths.append(rect_width)
        boxes.append(box)
    
    if center is not None:
        if widths:
            # Choose the region with the largest width (adjust as needed)
            idx = np.argmax(widths)
            best_box = boxes[idx]
            result = best_box
            reported_width = widths[idx]
        else:
            result = None
    else:
        result = np.mean(widths) if widths else None

    
   
    if center is not None:
        if result is not None:
            # Determine which rectangle edge is most vertical by checking x-coordinate differences.
            vertical_differences = [abs(result[i][0] - result[(i+1) % 4][0]) for i in range(4)]
            idx_edge = np.argmin(vertical_differences)
            p1 = result[idx_edge]
            p2 = result[(idx_edge+1)%4]
            # Also plot the opposite edge.
            opp_edge = (result[(idx_edge+2)%4], result[(idx_edge+3)%4])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='purple', linewidth=2)
            plt.plot([opp_edge[0][0], opp_edge[1][0]], [opp_edge[0][1], opp_edge[1][1]], color='purple', linewidth=2)
    else:
        # When no center is provided, plot vertical edges for all regions.
        for box in boxes:
            vertical_differences = [abs(box[i][0] - box[(i+1)%4][0]) for i in range(4)]
            idx_edge = np.argmin(vertical_differences)
            p1 = box[idx_edge]
            p2 = box[(idx_edge+1)%4]
            opp_edge = (box[(idx_edge+2)%4], box[(idx_edge+3)%4])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='purple', linewidth=2)
            plt.plot([opp_edge[0][0], opp_edge[1][0]], [opp_edge[0][1], opp_edge[1][1]], color='purple', linewidth=2)


    return result, reported_width



def measure_root_mask(root_mask, stalk_bbox):
    """
    Measure properties on a root mask.
    
    Parameters:
      root_mask: binary image (numpy array) where roots are white (non-zero)
      stalk_bbox: tuple (x, y, w, h) for the stalk bounding box
      
    Returns:
      dict with:
        - 'root_count': number of separated roots
        - 'avg_root_width': average width of individual roots
        - 'spread_width': horizontal distance between extreme root points
        - 'highest_emergence': (x, y) coordinate of the emergence spot
        - 'root_angle': angle (in degrees) between the stalk direction (vertical)
                        and the line from the spread width center to the emergence point
    """
    measurements = {}
    
    # Preprocess: noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(root_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area (dilate)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # For sure foreground, use distance transform and threshold.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Filter out unwanted contours: keep only the contour directly below the stalk_bbox.
    contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Unpack stalk_bbox assuming format (x, y, w, h)
    stalk_x, stalk_y, stalk_w, stalk_h = stalk_bbox if not isinstance(stalk_bbox, np.ndarray) \
        else (int(np.min(stalk_bbox[:,0])), int(np.min(stalk_bbox[:,1])), int(np.ptp(stalk_bbox[:,0])), int(np.ptp(stalk_bbox[:,1])))
    target_contours = []
    print(f"Found {len(contours)} contours.")
    for i,cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Check for horizontal overlap with stalk_bbox and that the contour lies below it
        print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")
        print(f"Stalk bbox: x={stalk_x}, y={stalk_y}, w={stalk_w}, h={stalk_h}")
        # if (x + w > stalk_x) and (x < stalk_x + stalk_w) and (y > stalk_y + stalk_h):
        if (x + w > stalk_x) and (x < stalk_x + stalk_w):
            print("Found target contour.")
            target_contours.append(cnt)
            # break
    print(f"Found {len(target_contours)} target contours.")
    if target_contours is not None:
        mask_filtered = np.zeros_like(root_mask)
        for cnt in target_contours:
            masklet = np.zeros_like(root_mask)
            cv2.drawContours(masklet, [cnt], -1, 255, thickness=-1)
            mask_filtered = cv2.bitwise_or(mask_filtered, masklet)
        # cv2.drawContours(mask_filtered, [target_contours], -1, 255, thickness=-1)
    else:
        mask_filtered = root_mask
    # Root spread width: horizontal distance between furthest root mask points.
    pts = np.column_stack(np.where(mask_filtered>0))
    if pts.size:
        # Note: pts are in (row, col) so col->x coordinate.
        xs = pts[:,1]
        measurements['spread_width'] = int(xs.max() - xs.min())
    else:
        measurements['spread_width'] = 0

    # Highest emergence spot:
    # Use the largest contour as the main root system.
    if target_contours:
        largest_contour = max(target_contours, key=cv2.contourArea)
        # Get the stalk vertical line. For instance, take left side of stalk_bbox.
        
        stalk_x = stalk_bbox[1][0] if stalk_bbox[1][0] < stalk_bbox[3][0] else stalk_bbox[3][0]
        # Find intersection of largest contour with vertical line (all contour points with x≈stalk_x)
        # Allow small error tolerance
        tol = 5
        emergence_candidates = [pt[0] for pt in largest_contour if abs(pt[0][0] - stalk_x) < tol]
        if emergence_candidates:
            # From these candidates, take the one with minimum y (highest point)
            emergence_points = [pt[0] for pt in largest_contour if abs(pt[0][0] - stalk_x) < tol]
            highest_pt = min(emergence_points, key=lambda p: p[1])
            measurements['highest_emergence'] = (int(highest_pt[0]), int(highest_pt[1]))
        else:
            measurements['highest_emergence'] = None
    else:
        measurements['highest_emergence'] = None

    # Root angle:
    # Compute the center along the spread width line
    if pts.size and measurements['highest_emergence']:
        center_x = (xs.max() + xs.min())/2
        
        y_mean = np.max(pts[:,0]) - (np.max(pts[:,0]) - np.median(pts[:,0]))/2  # approximate vertical center of roots, lowered by 10 pixels
        spread_center = (center_x, y_mean)
        emergence = measurements['highest_emergence']
        # Stalk line is vertical. The angle is computed from the vertical:
        measurements['spread_center'] = spread_center
        dx = emergence[0] - xs.min()
        dy = spread_center[1] - emergence[1]
        # Angle between vertical and the line. Use arctan of the horizontal offset versus vertical.
        print(f"dx: {dx}, dy: {dy}")
        angle_rad = math.atan2(dx, dy)
        measurements['root_angle'] = math.degrees(angle_rad)
    else:
        measurements['root_angle'] = None

    return measurements

def plot_measurements(image, bbox, measurements, mask=None, return_image=False):
    highest_emergence = measurements.get('highest_emergence')
    spread_center = measurements.get('spread_center')
    
    # Create a figure with two rows - top for measurements, bottom for masks
    fig = plt.figure(figsize=(15, 12))
    
    # Top plot: measurements overlay
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(image.permute(1, 2, 0), cmap='gray')
    ax1.set_title("Overlay: BBox, Highest Emergence, and Connection Line")
    ax1.axis('off')

    # Plot the bounding box by closing the polygon (bbox is numpy array of shape (4,2))
    bbox_closed = np.vstack([bbox, bbox[0]])
    ax1.plot(bbox_closed[:, 0], bbox_closed[:, 1], color='cyan', linewidth=2, label='BBox')

    # Plot highest emergence if it exists
    if highest_emergence is not None:
        ax1.scatter(highest_emergence[0], highest_emergence[1], c='red', s=80, label='Highest Emergence')

    # Plot the line from highest emergence to the spread_center if available
    if spread_center is not None and measurements.get('spread_width', 0) > 0:
        half_width = measurements['spread_width'] / 2
        ax1.hlines(spread_center[1],
                spread_center[0] - half_width,
                spread_center[0] + half_width,
                color='magenta',
                linestyle='-',
                linewidth=2,
                label='Spread Line')
        if highest_emergence is not None and spread_center is not None:
            # Draw a line from highest emergence to spread_center
            ax1.plot([highest_emergence[0], spread_center[0]- half_width], [highest_emergence[1], spread_center[1]], 
                color='yellow', linestyle='--', linewidth=2, label='Emergence-Line')
    ax1.legend(loc='upper right')
    
    # Bottom row: original image and masks
    if mask is not None:
        # Plot the original image
        plt.subplot(2, 5, 6)
        plt.imshow(image.permute(1, 2, 0))
        plt.title("Original Image")
        plt.axis("off")

        # Plot each mask channel
        for i in range(min(4, mask.shape[0])):  # Assuming mask has shape (C, H, W)
            plt.subplot(2, 5, i + 7)
            plt.imshow(mask[i].cpu().numpy(), cmap='gray')
            plt.title(f"Mask {i}")
            plt.axis("off")
    
    plt.tight_layout()
    
    if return_image:
        # Convert plot to image
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        # Convert ARGB to RGB
        buf = buf[:, :, 1:4]
        
        # Close the figure to free memory
        plt.close(fig)
        
        return buf
    else:
        plt.show()
        plt.close(fig)
        return None
    
def aprox_count_and_width(mask, image, stalk_bbox=None):
    """
    Estimate the number of roots and their average width in a root mask.
    
    Parameters:256 stalk ([], [], [], [])
    
    Returns:
      tuple (root_count, avg_width)
    """
    # ensure binary mask
    print(f"Mask shape: {mask.shape}")
    print(f"Image shape: {image.shape}")
    mask = mask.astype(np.uint8)
    #get contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if stalk_bbox is None:
        stalk_bbox = ([mask.shape[0]//3,0], [0,0], [mask.shape[0]//3,0], [69,69])
    for contour in contours: 
      x, y, w, h = cv2.boundingRect(contour)
      if ((x < stalk_bbox[0][0] and x+w > stalk_bbox[0][0]) or 
          (x < stalk_bbox[2][0] and x+w > stalk_bbox[2][0]) or 
          (x > stalk_bbox[0][0] and x+w < stalk_bbox[2][0])):
          cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    # Threshold the mask to ensure it is binary (0 or 255)
    # This helps in removing "ghost" artifacts if the mask has intermediate values.
    mask = (mask > 127).astype(np.uint8) * 255
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)
    # Blur the image to reduce noise and smooth out details
    image = cv2.GaussianBlur(image, (5, 5), 10)
    edges = cv2.Canny(image.astype(np.uint8), 0, 200) # Adjust thresholds as needed
    # Find contours in the edge-detected image
    edges = cv2.dilate(edges, None, iterations=1)
    # skeletonize the dilated edge image
    skeleton = skeletonize(edges > 0)
    # convert boolean skeleton to uint8 for plotting
    edges = (skeleton.astype(np.uint8)) * 255
    # Count the number of contours
    edges = cv2.bitwise_not(edges)  # Invert the edges to get the roots
    edges = cv2.bitwise_and(edges, mask)  # Apply the mask to the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #remove all contours smaller than the average contor area 
    print(f"Found {len(contours)} contours in the edge-detected image.")
    contour_areas = [cv2.contourArea(c) for c in contours]
    avg_area = np.mean(contour_areas)
    print(f"Average contour area: {avg_area:.2f}")
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    root_count = len(contours)
    root_widths = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate the width of the root
        root_width = w
        root_widths.append(root_width)
    # Calculate the average width of the roots
    if root_widths:
        avg_width = np.mean(root_widths)
    else:
        avg_width = 0
    return root_count, avg_width