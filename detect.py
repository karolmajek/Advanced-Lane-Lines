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
#
# # Edit this function to create your own pipeline.
# def pipeline(img, s_thresh=(100, 200), sx_thresh=(60, 100)):
#     img = np.copy(img)
#     # Convert to HSV color space and separate the V channel
#     # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     kernel = np.ones((5,5),np.float32)/25
#     img = cv2.filter2D(img,-1,kernel)
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
#     l = hsv[:,:,1]
#     s = hsv[:,:,1]
#     h = hsv[:,:,0]
#
#
#
#     # Sobel x
#     # sobelx = cv2.Laplacian(l_channel, cv2.CV_64F) # Take the derivative in x
#     sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0) # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#
#     # Threshold x gradient
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
#
#     # Threshold color channel
#     # lap = cv2.Laplacian(h, cv2.CV_64F) # Take the derivative in x
#     # abs_lap = np.absolute(lap) # Absolute x derivative to accentuate lines away from horizontal
#     # scaled_lap = np.uint8(255*abs_lap/np.max(abs_lap))
#     s_binary = np.zeros_like(h)
#     s_binary[(h >= s_thresh[0]) & (h <= s_thresh[1])] = 1
#
#     # s_binary = np.zeros_like(scaled_lap)
#     # s_binary[(scaled_lap >= s_thresh[0]) & (scaled_lap <= s_thresh[1])] = 1
#
#     # Stack each channel
#     # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
#     # be beneficial to replace this channel with something else.
#     # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
#     # return color_binary
#
#     result=sxbinary+s_binary
#     vertices=np.array([[[0,result.shape[0]],[result.shape[1],result.shape[0]],[result.shape[1]*0.5,result.shape[0]*0.55]]],dtype=np.int32)
#     result=region_of_interest(result*255, vertices)
#     # result[0:0.7*result.shape[0],:]=0
#
#     # cv2.imshow('scaled_sobel',scaled_sobel)
#     # cv2.imshow('s_binary',s_binary*255)
#     # cv2.imshow('sx_binary',sxbinary*255)
#     cv2.imshow('sxbinary+s_binary',result)
#     # cv2.imshow('h',h)
#     # cv2.imshow('sxbinary+s_binary',sxbinary+s_binary)
#     # cv2.waitKey(1)
#     return result/255.0


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(100, 200), sx_thresh=(60, 100)):
    img = np.copy(img)



    # Convert to HSV color space and separate the V channel
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    hsv[:,:,0] = cv2.equalizeHist(hsv[:,:,0])

    img_wht = cv2.cvtColor(hsv, cv2.COLOR_YUV2BGR)
    img_ylw = cv2.cvtColor(img_wht, cv2.COLOR_BGR2HSV)
    img_wht=img_wht[:,:,1]
    img_wht[img_wht<255]=0

    mask_wht = cv2.inRange(img_wht, 255, 255)

    lower_ylw = np.array([60,135,50])
    upper_ylw = np.array([180,155,155])
    # img_ylw[img_ylw<(lower_ylw)]=0
    # img_ylw[img_ylw>(upper_ylw)]=0
    # lower_wht = np.array([0, 130, 125])
    # upper_wht = np.array([255,135,130])

    # mask_wht = cv2.inRange(hsv, lower_wht,upper_wht)
    mask_ylw = cv2.inRange(hsv, lower_ylw, upper_ylw)

    mask = mask_wht+mask_ylw

    return mask

    res = cv2.bitwise_and(img,img, mask= mask)

    hsv = hsv.astype(np.float)


    cv2.imshow('result',np.concatenate((res,img_ylw),axis=1))
    # cv2.imshow('result',img_ylw)
    cv2.waitKey(1)


    # h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    # s1= cv2.adaptiveThreshold(s.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)





    #
    # v=s
    # v[v>210]=255
    # v[v<80]=80
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # v = clahe.apply(v.astype(np.uint8)).astype(np.float)

    # cv2.imshow('s,v',np.concatenate((s/255.0,v/255.0)))

    sobel_s = cv2.Sobel(s, cv2.CV_64F, 1, 0)
    abs_sobel_s=sobel_s
    abs_sobel_s[abs_sobel_s<0] = 0
    abs_sobel_s[s<150] = 0
    # scaled_sobel_s = abs_sobel_s
    scaled_sobel_s = 255.0*abs_sobel_s/np.max(abs_sobel_s)
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobel_v = cv2.Sobel(v, cv2.CV_64F, 1, 0)
    abs_sobel_v=sobel_v
    abs_sobel_v[abs_sobel_v<0] = 0
    # abs_sobel_v = np.absolute(sobel_v)
    scaled_sobel_v = 255.0*abs_sobel_v/np.max(abs_sobel_v)

    #
    # lap_s = cv2.Laplacian(s, cv2.CV_64F)
    # abs_lap_s =lap_s
    # abs_lap_s[abs_lap_s<0] =0
    # # abs_lap_s = np.absolute(lap_s)
    # # scaled_lap_s = abs_lap_s
    # scaled_lap_s = 255.0*abs_lap_s/np.max(abs_lap_s)
    #
    # lap_v = cv2.Laplacian(v, cv2.CV_64F)
    # abs_lap_v = np.absolute(lap_v)
    # scaled_lap_v = 255.0*abs_lap_v/np.max(abs_lap_v)

    # s1=s
    v1=v
    # s1[s1<180]=0
    # s1[s1>=180]=255

    blur = cv2.GaussianBlur(scaled_sobel_s.astype(np.uint8),(5,5),0)
    _,s1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(scaled_sobel_v.astype(np.uint8),(5,5),0)
    _,v1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # s1= cv2.adaptiveThreshold(s.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    v1[v1<150]=0
    v1[v1>=150]=255

    res=s1
    res[v1>200]=255

    # cv2.imshow('s,v',np.concatenate((np.concatenate((scaled_sobel_s/255.0,scaled_sobel_v/255.0)),np.concatenate((s1/255.0,v1/255.0))), axis=1))
    # cv2.imshow('res',res)

    return res/255.0

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
    # process('CarND-Advanced-Lane-Lines/project_video.mp4')
    process('CarND-Advanced-Lane-Lines/challenge_video.mp4')
    process('CarND-Advanced-Lane-Lines/harder_challenge_video.mp4')
def process(fname):
    is_distortion_saved=False

    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(fname)

    old_l_x=[]
    old_l_y=[]
    old_r_x=[]
    old_r_y=[]

    while(cap.isOpened()):

        for ttt in range(1):
            ret, img = cap.read()

        if img is None:
            break
        img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
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


        corners=[[0.16*undist.shape[1],undist.shape[0]],
                [0.45*undist.shape[1],0.63*undist.shape[0]],
                [0.55*undist.shape[1],0.63*undist.shape[0]],
                [0.84*undist.shape[1],undist.shape[0]]]
        src = np.float32(corners)

        # offset=(warped_size[0]-warped_size[1])/2.0
        offset=200
        warped_size=(600+2*offset,600)
        dst = np.float32([
                            [offset, warped_size[1]],
                            [offset, 0],
                            [offset+warped_size[1], 0],
                            [offset+warped_size[1], warped_size[1]]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped_orig = cv2.warpPerspective(undist, M, dsize=warped_size)
        warped=pipeline(warped_orig, s_thresh=(50,90))

        pts = np.array(src, np.int32)
        pts = pts.reshape((-1,1,2))
        # cv2.polylines(undist,[pts],True,(0,0,255))

        kernel = np.ones((3,3),np.uint8)
        warped = cv2.erode(warped,kernel,iterations = 2)
        warped = cv2.dilate(warped,kernel,iterations = 2)



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
        # for line in warped[warped.shape[0]/2:,:]:
            # print(line)
        # print(points)
        # print(histogram)
        # ax3.plot(histogram)
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


        # kernel = np.ones((5,5),np.uint8)
        # erosion = cv2.erode(warped,kernel,iterations = 1)
        # erosion = cv2.erode(erosion,kernel,iterations = 1)
        # warped = cv2.erode(erosion,kernel,iterations = 1)


        if len(points[0]) == 0 or len(points[1])==0:
            continue

        leftx,lefty= zip(*points[0])
        rightx,righty= zip(*points[1])

        all_y=np.array(list(range(warped.shape[0])))

        # tl_x=list(leftx)+old_l_x
        # tl_y=list(lefty)+old_l_y
        # tr_x=list(rightx)+old_r_x
        # tr_y=list(righty)+old_r_y


        tl_x=list(leftx)
        tl_y=list(lefty)
        tr_x=list(rightx)
        tr_y=list(righty)

        old_l_x=list(leftx)
        old_l_y=list(lefty)
        old_r_x=list(rightx)
        old_r_y=list(righty)

        # leftx=np.array(leftx)
        # lefty=np.array(lefty)
        # rightx=np.array(rightx)
        # righty=np.array(righty)

        tl_x=np.array(tl_x)
        tl_y=np.array(tl_y)
        tr_x=np.array(tr_x)
        tr_y=np.array(tr_y)
        # Fit a second order polynomial to each fake lane line
        left_fit = np.array(np.polyfit(tl_y, tl_x, 2))
        left_fitx = np.array(left_fit[0]*all_y**2 + left_fit[1]*all_y + left_fit[2])
        right_fit = np.array(np.polyfit(tr_y, tr_x, 2))
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
        result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)
        # plt.axes(ax4)
        # plt.imshow(result)



        unwarped = 255*cv2.warpPerspective(warped, Minv, (img.shape[1], img.shape[0])).astype(np.uint8)
        unwarped=np.dstack((np.zeros_like(unwarped),np.zeros_like(unwarped),unwarped))
        # print(undist.dtype)
        # print(unwarped.dtype)
        # print('-'*30)
        line_image = cv2.addWeighted(result, 1, unwarped, 1, 0)



        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(all_y)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                     /np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                        /np.absolute(2*right_fit[0])
        print(left_curverad, right_curverad)

        result=cv2.putText(result,'%.1f %.1f'%(left_curverad, right_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)


        # cv2.imshow('warped',warped)
        # hls_binary = cv2.cvtColor((hls_binary*255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
        # print(hls_binary.shape)
        # print(result.shape)

        # cv2.imshow('result',line_image)
        cv2.imshow('result',np.concatenate((line_image,result), axis=1))
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
