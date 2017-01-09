#!/usr/bin/python3
from sklearn.cluster import KMeans
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
# images=[images[3]]

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
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary
    return sxbinary+s_binary

def getCenters(histogram,threshold=10):
    x_val_hist=[i for i,x in enumerate(histogram) if x>threshold]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(x_val_hist).reshape(-1,1))
    return kmeans.cluster_centers_

def findLinePoints(image,centers):
    points=([],[])
    for i,c in enumerate(centers):
        for y in range(image.shape[0]):
            for x in range(int(c[0])-30,int(c[0])+30):
                if x<image.shape[1] and y<image.shape[0]:
                    if image[y][x]!=0:
                        points[i].append((x,y))
    return points

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


        f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 9))
        f.tight_layout()
        # ax1.imshow(hls_binary, cmap='gray')
        # ax1.imshow(hls_binary)
        ax1.imshow(undist)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped, cmap='gray')
        # ax2.imshow(hls_binary, cmap='gray')
        ax2.set_title('Thresholded S', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()
        histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)

        centers=getCenters(histogram)
        print(centers)
        points=findLinePoints(warped,centers)
        # print(points)
        # print(histogram)
        ax3.plot(histogram)
        # plt.show()
        # Generate some fake data to represent lane-line pixels
        # yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
        # leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        #                               for idx, elem in enumerate(yvals)])
        # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        # rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        #                                 for idx, elem in enumerate(yvals)])
        # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
        # left_fit = np.polyfit(yvals, leftx, 2)
        # left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
        # right_fit = np.polyfit(yvals, rightx, 2)
        # right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
        # # Plot up the fake data
        # plt.plot(leftx, yvals, 'o', color='red')
        # plt.plot(rightx, yvals, 'o', color='blue')
        # plt.xlim(0, 1280)
        # plt.ylim(0, 720)
        # plt.plot(left_fitx, yvals, color='green', linewidth=3)
        # plt.plot(right_fitx, yvals, color='green', linewidth=3)
        # plt.gca().invert_yaxis() # to visualize as we do the images
        # plt.show()

        leftx,lefty= zip(*points[0])
        rightx,righty= zip(*points[1])

        leftx=np.array(leftx)
        lefty=np.array(lefty)
        rightx=np.array(rightx)
        righty=np.array(righty)
        # Fit a second order polynomial to each fake lane line
        left_fit = np.array(np.polyfit(lefty, leftx, 2))
        left_fitx = np.array(left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2])
        right_fit = np.array(np.polyfit(righty, rightx, 2))
        right_fitx = np.array(right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2])


        plt.axes(ax4)
        # Plot up the fake data
        plt.plot(leftx, lefty, 'o', color='red')
        plt.plot(rightx, righty, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, lefty, color='green', linewidth=3)
        plt.plot(right_fitx, righty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()
        # return



if __name__ == '__main__':
    main()
