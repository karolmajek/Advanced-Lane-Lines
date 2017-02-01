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

class LaneDetector:
    """
    My class for lane detetion.
    """
    def mask_lane_lines(self,img, s_thresh=(100, 200), sx_thresh=(60, 100)):
        img = np.copy(img)

        #blur
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)

        #hsv
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

    # def getCenters(self,warped,threshold=10):
    #     histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
    #
    #     x_val_hist=[i for i,x in enumerate(histogram) if x>threshold]
    #     kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(x_val_hist).reshape(-1,1))
    #     return histogram,kmeans.cluster_centers_

    def findLinePoints(self,image):
        points=([],[])
        for yy,line in enumerate(image):
            if np.count_nonzero(line)>10:
                x_val_hist=[]
                counter=0
                for x in line:
                    if x>0:
                        x_val_hist.append(counter)
                    counter=counter+1
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
        return points

    def plotFitted(self,yvals,leftx,rightx,left_fitx,right_fitx):
        # # Generate some fake data to represent lane-line pixels
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
        # Plot up the fake data
        # plt.plot(leftx, yvals, 'o', color='red')
        # plt.plot(rightx, yvals, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, yvals, color='green', linewidth=3)
        plt.plot(right_fitx, yvals, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()
    def process(self,fname):
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

            #remove car
            undist[-int(undist.shape[0]*0.1):-1,:,:]=0
            #remove sky
            undist[0:int(undist.shape[0]*0.5),:,:]=0

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
            warped=self.mask_lane_lines(warped_orig, s_thresh=(50,90))

            pts = np.array(src, np.int32)
            pts = pts.reshape((-1,1,2))
            # cv2.polylines(undist,[pts],True,(0,0,255))

            kernel = np.ones((3,3),np.uint8)
            warped = cv2.erode(warped,kernel,iterations = 2)
            warped = cv2.dilate(warped,kernel,iterations = 2)


            points=self.findLinePoints(warped)


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

            # self.plotFitted(all_y,leftx,rightx,left_fitx,right_fitx)
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


            cv2.imshow('result',np.concatenate((line_image,result), axis=1))
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

def main():
    det=LaneDetector()
    det.process('CarND-Advanced-Lane-Lines/project_video.mp4')
    det.process('CarND-Advanced-Lane-Lines/challenge_video.mp4')
    det.process('CarND-Advanced-Lane-Lines/harder_challenge_video.mp4')

if __name__ == '__main__':
    main()
