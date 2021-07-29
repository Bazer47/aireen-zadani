# -*- coding: utf-8 -*-
"""
@author: Jan Bayer
"""
import json
import numpy as np
import cv2 as cv
from matplotlib import colors
import copy

def get_color(color_name: str) -> list:
    ''' Get RGB color from color name
    '''
    return [x*255 for x in colors.to_rgb(color_name)][::-1]

def resize_image(img_path: str, json_path: str, target_dim: tuple,
                 border_top = 25):
    '''
    Takes image, resizes it, adds annotations from json and creates
    an updated json.
    
    Parameters
    ----------
    img_path : str
        Path of the default image.
    json_path : str
        Path of the default json.
    target_dim : tuple
        In the openCV format (width, height).
    border_top : int, optional
        Space between the new retina image and the target image edge.
        The default is 25.

    Returns
    -------
    target_img_clean : array
        Target image without annotations.
    target_img : array
        Target image with annotations.
    new_json : dict
        Json dict with new/resized coordinates.
    '''
    # Load def_img and def_json
    f = open(json_path)
    def_json = json.load(f)
    f.close()
    def_img = cv.imread(img_path)
    
    # Create copy of the default json
    new_json = copy.deepcopy(def_json)
    
    # Extract retina
    # BGR to Gray and to black/white image
    gray = cv.cvtColor(def_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Create bounding rectangle and crop
    x,y,w,h = cv.boundingRect(thresh)
    cropped = def_img[y:y+h, x:x+w] 
    
    # Scaling coeff for the retina crop = target/default
    scale_coeff_retina = (target_dim[1] - 2*border_top)/cropped.shape[0] 
    
    # Resize the retina crop
    # Keeps the aspect ratio
    resized_img_retina = cv.resize(cropped,
                                   None,
                                   fx=scale_coeff_retina,
                                   fy=scale_coeff_retina,
                                   interpolation=cv.INTER_LINEAR)
    
    # Create border around resized retina crop to obtain target img
    border_right = int((target_dim[0] - resized_img_retina.shape[1])/2)
    target_img = cv.copyMakeBorder(
                         resized_img_retina, 
                         border_top, 
                         border_top, 
                         border_right, 
                         border_right, 
                         cv.BORDER_CONSTANT, 
                         value=0
                         )
    assert ((target_dim[0] == target_img.shape[1])
            and (target_dim[1] == target_img.shape[0])),\
            'Target img dimensions are not equal to the target_dim'
    
    target_img_clean = target_img.copy()
    
    # Update and plot annotations
    # Iterate through segments/lesions
    font = cv.FONT_HERSHEY_SIMPLEX
    seg_objects = def_json['segmentedObjectDict']
    
    for seg_object in seg_objects:   
        
        pts = np.array(seg_objects[seg_object]['pointsList']).reshape((-1,1,2))
        pts_color = seg_objects[seg_object]['ClassColorName']
        pts_name = seg_objects[seg_object]['Name']
        
        # Transform points to new coordinates
        # 1) Moves; 2) Scales; 3) Moves coord. sys.
        move_points_1 = np.broadcast_to((x, y), pts.shape)
        move_points_2 = np.broadcast_to((border_right, border_top), pts.shape)
        
        pts_resized = (((pts
                         - move_points_1) 
                        * scale_coeff_retina) 
                       + move_points_2).astype(np.int32)
        
        # Create the area from points
        cv.polylines(target_img,
                     [pts_resized],
                     True,
                     get_color(pts_color))
        # Add text
        idx = np.where(pts_resized[:,0,1]==np.amax(pts_resized[:,0,1]))[0][0]
        text_pos = pts_resized[idx,0,:] + np.array((2, 10))
        cv.putText(target_img,
                   pts_name,
                   text_pos,
                   font,
                   0.15,
                   get_color(pts_color),
                   1,
                   cv.LINE_AA)
        
        # Store pts_resized back in dict
        new_json['segmentedObjectDict'][seg_object]['pointsList'] =\
            pts_resized.squeeze().tolist()
        
    return target_img_clean, target_img, new_json

if __name__ == "__main__":
    
    DATA_PATH = 'data/'
    
    target_img_clean, target_img,\
        new_json = resize_image(DATA_PATH + 'testmaxresdefault.jpg',
                                DATA_PATH + 'testmaxresdefault.json',
                                (866,574))
     
    # Store the json and images
    with open('target_img.json', 'w') as fp:
        json.dump(new_json, fp, indent=4)
    cv.imwrite('target_img_clean.jpg', target_img_clean)
    cv.imwrite('target_img.jpg', target_img)