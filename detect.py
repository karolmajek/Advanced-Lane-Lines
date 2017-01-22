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

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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
def pipeline(img, s_thresh=(100, 200), sx_thresh=(60, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l = hsv[:,:,1]
    s = hsv[:,:,1]
    h = hsv[:,:,0]



    # Sobel x
    # sobelx = cv2.Laplacian(l_channel, cv2.CV_64F) # Take the derivative in x
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    # lap = cv2.Laplacian(h, cv2.CV_64F) # Take the derivative in x
    # abs_lap = np.absolute(lap) # Absolute x derivative to accentuate lines away from horizontal
    # scaled_lap = np.uint8(255*abs_lap/np.max(abs_lap))
    s_binary = np.zeros_like(h)
    s_binary[(h >= s_thresh[0]) & (h <= s_thresh[1])] = 1

    # s_binary = np.zeros_like(scaled_lap)
    # s_binary[(scaled_lap >= s_thresh[0]) & (scaled_lap <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary

    result=sxbinary+s_binary
    vertices=np.array([[[0,result.shape[0]],[result.shape[1],result.shape[0]],[result.shape[1]*0.5,result.shape[0]*0.55]]],dtype=np.int32)
    result=region_of_interest(result*255, vertices)
    # result[0:0.7*result.shape[0],:]=0

    # cv2.imshow('scaled_sobel',scaled_sobel)
    # cv2.imshow('s_binary',s_binary*255)
    # cv2.imshow('sx_binary',sxbinary*255)
    cv2.imshow('sxbinary+s_binary',result)
    # cv2.imshow('h',h)
    # cv2.imshow('sxbinary+s_binary',sxbinary+s_binary)
    # cv2.waitKey(1)
    return result/255.0

def getCenters(warped,threshold=10):
    # for line in warped:
    #     if np.count_nonzero(line)>10:
    #         x_val_hist=[i for i,x in enumerate(line) if x>0]
    #         kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(x_val_hist).reshape(-1,1))
    #         means=list(kmeans.cluster_centers_.reshape(2))
    #         if np.abs(means[0]-means[1])>300:
    #             print(np.count_nonzero(line),list(kmeans.cluster_centers_.reshape(2)))
    histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)

    x_val_hist=[i for i,x in enumerate(histogram) if x>threshold]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(x_val_hist).reshape(-1,1))
    return histogram,kmeans.cluster_centers_

# @profile
def findLinePoints(image,centers=None,fit=None):
    points=([],[])
    # if fit is None:
    #     for i,c in enumerate(centers):
    #         for y in range(image.shape[0]):
    #             for x in range(int(c[0])-100,int(c[0])+100):
    #                 if x<image.shape[1] and y<image.shape[0]:
    #                     if image[y][x]!=0:
    #                         points[i].append((x,y))
    # else:
    for yy,line in enumerate(image):
        if np.count_nonzero(line)>10:
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            # x_val_hist=[i for i,x in enumerate(line) if x>0]
            # x_val_hist=[x[0] for x in zip(list(range(len(line))),line) if x[1]>0]
            # print(x_val_hist)
            xmin=np.min(np.array(x_val_hist))
            xmax=np.max(np.array(x_val_hist))
            if np.abs(xmax-xmin)>300:
                center=(xmin+xmax)/2.0
                left=[(x,yy) for x in x_val_hist if x<center]
                right=[(x,yy) for x in x_val_hist if x>=center]
                for l in left:
                    points[0].append(l)
                for r in right:
                    points[1].append(r)
            # kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(x_val_hist).reshape(-1,1))
            # means=list(kmeans.cluster_centers_.reshape(2))
            # if np.abs(means[0]-means[1])>300:
            #     print(np.count_nonzero(line),list(kmeans.cluster_centers_.reshape(2)))
            #     for i in list(kmeans.cluster_centers_.reshape(2)):
            #         if i<500:
            #             points[0].append((i,yy))
            #         else:
            #             points[1].append((i,yy))

    return points


def main():
    is_distortion_saved=False

    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture('CarND-Advanced-Lane-Lines/project_video.mp4')
    # cap = cv2.VideoCapture('CarND-Advanced-Lane-Lines/challenge_video.mp4')
    # cap = cv2.VideoCapture('CarND-Advanced-Lane-Lines/harder_challenge_video.mp4')

    while(cap.isOpened()):
        ret, img = cap.read()
        if img is None:
            break
    # for im in images:
    #     img=cv2.imread(im)
        undist = cv2.undistort(img,calib['matrix'], calib['dist'], None,None)
        # cv2.imshow('undistorted',dst)
        # cv2.waitKey(0)

        #Save before and after undistortion
        if not is_distortion_saved:
            dist_before_after=np.concatenate((img,undist), axis=1)
            dist_before_after=cv2.putText(dist_before_after,'Distorted',(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            dist_before_after=cv2.putText(dist_before_after,'Undistorted',(50+img.shape[1],50), font, 1,(255,255,255),2,cv2.LINE_AA)

            cv2.imwrite('images/distortion.jpg',dist_before_after)
            is_distortion_saved=True

        # hls_binary = hls_select(dst, thresh=(150, 250))
        hls_binary=pipeline(undist, s_thresh=(50,90))

        corners=[[0.16*undist.shape[1],undist.shape[0]],
                [0.45*undist.shape[1],0.63*undist.shape[0]],
                [0.55*undist.shape[1],0.63*undist.shape[0]],
                [0.84*undist.shape[1],undist.shape[0]]]
        src = np.float32(corners)

        # offset=(warped_size[0]-warped_size[1])/2.0
        offset=100
        warped_size=(600+2*offset,600)
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



        # f, ((ax1, ax2,ax3),(ax3,ax4,ax5)) = plt.subplots(2, 3, figsize=(16, 12))
        # # f.tight_layout()
        # # ax1.imshow(hls_binary, cmap='gray')
        # # ax1.imshow(hls_binary)
        # ax1.imshow(undist[::2,::2,:])
        # # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(warped, cmap='gray')
        # # ax2.imshow(hls_binary, cmap='gray')
        # # ax2.set_title('Thresholded S', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # # plt.show()

        # histogram,centers=getCenters(warped)
        # print('Centers:',centers)
        # points=findLinePoints(warped,centers,1)
        points=findLinePoints(warped)
        # # for line in warped[warped.shape[0]/2:,:]:
        #     # print(line)
        # # print(points)
        # # print(histogram)
        # ax3.plot(histogram)
        # # plt.show()
        # # Generate some fake data to represent lane-line pixels
        # # yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
        # # leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        # #                               for idx, elem in enumerate(yvals)])
        # # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        # # rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        # #                                 for idx, elem in enumerate(yvals)])
        # # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
        # # left_fit = np.polyfit(yvals, leftx, 2)
        # 1# left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
        # # right_fit = np.polyfit(yvals, rightx, 2)
        # # right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
        # # # Plot up the fake data
        # # plt.plot(leftx, yvals, 'o', color='red')
        # # plt.plot(rightx, yvals, 'o', color='blue')
        # # plt.xlim(0, 1280)
        # # plt.ylim(0, 720)
        # # plt.plot(left_fitx, yvals, color='green', linewidth=3)
        # # plt.plot(right_fitx, yvals, color='green', linewidth=3)
        # # plt.gca().invert_yaxis() # to visualize as we do the images
        # # plt.show()

        if len(points[0]) == 0 or len(points[1])==0:
            continue

        leftx,lefty= zip(*points[0])
        rightx,righty= zip(*points[1])

        all_y=np.array(list(range(warped.shape[0])))

        leftx=np.array(leftx)
        lefty=np.array(lefty)
        rightx=np.array(rightx)
        righty=np.array(righty)
        # Fit a second order polynomial to each fake lane line
        left_fit = np.array(np.polyfit(lefty, leftx, 2))
        left_fitx = np.array(left_fit[0]*all_y**2 + left_fit[1]*all_y + left_fit[2])
        right_fit = np.array(np.polyfit(righty, rightx, 2))
        right_fitx = np.array(right_fit[0]*all_y**2 + right_fit[1]*all_y + right_fit[2])

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, all_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, all_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv=np.linalg.inv(M)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        # plt.axes(ax4)
        # plt.imshow(result)



        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(all_y)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                     /np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                        /np.absolute(2*right_fit[0])
        print(left_curverad, right_curverad)

        result=cv2.putText(result,'%.1f %.1f'%(left_curverad, right_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)


        cv2.imshow('result',result)
        cv2.waitKey(1)


        # plt.axes(ax5)
        # # Plot up the fake data
        # plt.plot(leftx, lefty, 'o', color='red')
        # plt.plot(rightx, righty, 'o', color='blue')
        # # plt.xlim(0, warped.shape[1])
        # # plt.ylim(0, warped.shape[0])
        # plt.plot(left_fitx, all_y, color='green', linewidth=3)
        # plt.plot(right_fitx, all_y, color='green', linewidth=3)
        # plt.gca().invert_yaxis() # to visualize as we do the images
        # plt.show()
        # return
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
