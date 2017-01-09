#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import json


calib=dict()
with open('calib.json', 'r') as f:
    calib=json.load(f)
calib['matrix']=np.array(calib['matrix'])
calib['dist']=np.array(calib['dist'])
# print(json.dumps(calib, indent=2))
# print('Calibration mat:',calib['matrix'])

images=glob.glob('CarND-Advanced-Lane-Lines/test_images/*.jpg')
images=[images[3]]

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,2]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

def main():
    for im in images:
        img=cv2.imread(im)
        undist = cv2.undistort(img,calib['matrix'], calib['dist'], None,None)
        # cv2.imshow('undistorted',dst)
        # cv2.waitKey(0)

        # hls_binary = hls_select(dst, thresh=(150, 250))
        hls_binary=pipeline(undist)

        corners=[[0.16*undist.shape[1],undist.shape[0]],
                [0.45*undist.shape[1],0.63*undist.shape[0]],
                [0.55*undist.shape[1],0.63*undist.shape[0]],
                [0.84*undist.shape[1],undist.shape[0]]]
        src = np.float32(corners)

        warped_size=(undist.shape[1],600)
        offset=(warped_size[0]-warped_size[1])/2.0
        dst = np.float32([
                            [offset, warped_size[1]],
                            [offset, 0],
                            [offset+warped_size[1], 0],
                            [offset+warped_size[1], warped_size[1]]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(hls_binary, M, dsize=warped_size)

        pts = np.array(src, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(undist,[pts],True,(0,0,255))


        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        # ax1.imshow(hls_binary, cmap='gray')
        # ax1.imshow(hls_binary)
        ax1.imshow(undist)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped)
        # ax2.imshow(hls_binary, cmap='gray')
        ax2.set_title('Thresholded S', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        # return



if __name__ == '__main__':
    main()
