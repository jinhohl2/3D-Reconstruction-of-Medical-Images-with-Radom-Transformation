import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale,resize,iradon
import scipy
import scipy.fftpack
from scipy.interpolate import griddata
from time import time
import nibabel as nib
import cv2
import os
import time

def change_range(img, in_min, in_max, out_min, out_max):
    in_range = in_max - in_min
    out_range = out_max - out_min
    scaled = np.array((img - in_min) / float(in_range), dtype=float)
    return out_min + (scaled * out_range)

def createPointCloud(filename, arr):
    # open file and write boilerplate header
    file = open(filename, 'w');
    file.write("ply\n");
    file.write("format ascii 1.0\n");

    # count number of vertices
    num_verts = arr.shape[0];
    file.write("element vertex " + str(num_verts) + "\n");
    file.write("property float32 x\n");
    file.write("property float32 y\n");
    file.write("property float32 z\n");
    file.write("property uchar red\n");
    file.write("property uchar green\n");
    file.write("property uchar blue\n");
    file.write("property uchar alpha\n");
    
    file.write("end_header\n");

    # write points
    point_count = 0;
    for point in arr:
        # progress check
        point_count += 1;
        if point_count % 1000 == 0:
            print("Point: " + str(point_count) + " of " + str(len(arr)));

        # create file string
        out_str = "";
        for axis in point:
            out_str += str(axis) + " ";
        out_str = out_str[:-1]; # dump the extra space
        out_str += '0' + " ";
        out_str += '0' + " ";
        out_str += '255' + " ";
        out_str += '255';
        out_str += "\n";
        file.write(out_str);
    file.close();

def addPoints(mask, points_list, depth):
    mask_points = np.where(mask == 255);
    for ind in range(len(mask_points[0])):
        # get point
        x = mask_points[1][ind];
        y = mask_points[0][ind];
        point = [x,y,depth];
        points_list.append(point);

if __name__ == '__main__':
    #nii = nib.load('archive/ct_scans/coronacases_org_001.nii').get_fdata()
    nii = np.loadtxt("nii_reconstructed.txt").reshape(300,300,301)

    #nii = change_range(nii, -1021 , 2996 ,0, 255)
    nii = change_range(nii, 
    -0.024124923553181857 , 1 ,0, 255)
    nii = np.uint8(nii)

    images = []
    for i in range(nii.shape[2]):
        image = nii[:,:,i]
        image = cv2.resize(nii[:,:,i],(300,300))
        images.append(image)
        
    blank_mask = np.zeros_like(images[0],np.uint8)
    blank_mask.shape

    masks = [];
    masks.append(blank_mask);
    for image in images:
        #mask = image
        mask = np.squeeze(image*[50 <= image]*[image <= 200])
        #mask = cv2.inRange(image, 0, 100); #best so far

        # show
        #cv2.imshow("Mask", mask);
        #cv2.waitKey(1);
        masks.append(mask);
    masks.append(blank_mask);
    #cv2.destroyAllWindows();

    depth = 0;
    points = [];
    slice_thickness = .2; # distance between slices
    xy_scale = 1;
    for index in range(1,len(masks)-1):
        # progress check
        print("Index: " + str(index) + " of " + str(len(masks)));

        # get three masks
        prev = masks[index - 1];
        curr = masks[index];
        after = masks[index + 1];

        # do a slice on previous
        prev_mask = np.zeros_like(curr);
        prev_mask[prev == 0] = curr[prev == 0];
        addPoints(prev_mask, points, depth);

        # # do a slice on after
        next_mask = np.zeros_like(curr);
        next_mask[after == 0] = curr[after == 0];
        addPoints(next_mask, points, depth);
        
        # get contour points (_, contours) in OpenCV 2.* or 4.*
        contours, _ = cv2.findContours(curr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);
        for con in contours:
            for point in con:
                p = point[0]; # contours have an extra layer of brackets
                points.append([p[0], p[1], depth]);

        # increment depth
        depth += slice_thickness;

    # rescale x,y points
    for ind in range(len(points)):
        # unpack
        x,y,z = points[ind];

        # scale
        x *= xy_scale;
        y *= xy_scale;
        points[ind] = [x,y,z];

    # convert points to numpy and dump duplicates
    points = np.array(points).astype(np.float32);
    points = np.unique(points.reshape(-1, points.shape[-1]), axis=0);
    print(points.shape);

    createPointCloud("3d_full_reconstructed_python.ply", points);
