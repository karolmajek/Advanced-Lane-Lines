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
    is_distortion_saved=False
    font = cv2.FONT_HERSHEY_SIMPLEX

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
        img_wht[img_wht<250]=0

        mask_wht = cv2.inRange(img_wht, 250, 255)

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
        shape=image.shape
        img=cv2.resize(image,(shape[1],int(shape[0])),fx=0,fy=0)
        red=np.zeros_like(img)
        green=np.zeros_like(img)
        blue=np.zeros_like(img)

        center=int(shape[1]/2)
        print(shape)
        for yy,line in list(enumerate(image))[::-1]:
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            if len(x_val_hist)>0:
                # xmin=np.min(np.array(x_val_hist))
                # xmax=np.max(np.array(x_val_hist))
                # if np.abs(xmax-xmin)>100:
                cv2.circle(green,(int(center),yy),1,(255,255,255))

                left=[(x,yy) for x in x_val_hist if x<center]
                right=[(x,yy) for x in x_val_hist if x>=center]
                if len(left)>0:
                    l=np.mean(np.array(left),axis=0)
                    l=(l[0],l[1])
                    print('center',center,'yy',yy,'l',l[0])
                    center=l[0]+int(shape[1]*0.2)
                    # new_center=l[0]+int(shape[1]*0.2)
                    # center=(center+new_center)*0.5
                    cv2.circle(red,(int(l[0]),int(l[1])),1,(255,255,255))
                    points[0].append(l)
                if len(right)>0:
                    r=np.mean(np.array(right),axis=0)
                    r=(r[0],r[1])
                    # print(r)
                    cv2.circle(blue,(int(r[0]),int(r[1])),1,(255,255,255))
                    points[1].append(r)
        img=cv2.resize(np.dstack((blue,green,red)),(shape[1],int(shape[0])),fx=0,fy=0)
        cv2.imshow('small',img)
        return points

    def findLinePointsFast(self,image):
        points=([],[])
        shape=image.shape
        # img=cv2.resize(image,(shape[1],int(shape[0]/10)),fx=0,fy=0)
        img=cv2.resize(image,(shape[1],int(shape[0])),fx=0,fy=0)
        red=np.zeros_like(img)
        blue=np.zeros_like(img)

        diff=self.right_fitx[0]-self.left_fitx[0]

        for y,lx,rx,line in list(zip(self.all_y,self.left_fitx,self.right_fitx,image))[::10]:
            lxmin=int(lx-20-0.2*(img.shape[0]-y))
            lxmax=int(lx+20+0.2*(img.shape[0]-y))
            rxmin=int(rx-20-0.2*(img.shape[0]-y))
            rxmax=int(rx+20+0.2*(img.shape[0]-y))
            cv2.circle(red,(lxmin,int(y)),1,(255,255,255))
            # cv2.circle(red,(int(lx),int(y)),1,(255,255,255))

            cv2.circle(red,(lxmax,int(y)),1,(255,255,255))
            cv2.circle(red,(rxmin,int(y)),1,(255,255,255))
            # cv2.circle(red,(int(rx),int(y)),1,(255,255,255))
            cv2.circle(red,(rxmax,int(y)),1,(255,255,255))
            # if np.count_nonzero(line)>10:
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            # if np.abs(xmax-xmin)>300:
            if len(x_val_hist)>5:
                xmin=np.min(np.array(x_val_hist))
                xmax=np.max(np.array(x_val_hist))
                # center=(xmin+xmax)/2.0
                left=[(x,y) for x in x_val_hist if x<=lxmax and x>=lxmin]
                right=[(x,y) for x in x_val_hist if x>=rxmin and x<=rxmax]
                l=None
                r=None
                if len(left):
                    l=np.mean(np.array(left),axis=0)
                if len(right):
                    r=np.mean(np.array(right),axis=0)
                if l==None or r==None or r[0]-l[0]>200:
                    if l!=None and l[0]>lxmin and l[0]<lxmax:
                        cv2.circle(blue,(int(l[0]),int(l[1])),1,(255,255,255))
                        points[0].append(l)
                        # if y<10:
                        #     points[1].append((l[0]+diff,l[1]))
                    if r!=None and r[0]>rxmin and r[0]<rxmax:
                        cv2.circle(blue,(int(r[0]),int(r[1])),1,(255,255,255))
                        points[1].append(r)

        img=cv2.resize(np.dstack((blue,red,red)),(shape[1],int(shape[0])),fx=0,fy=0)
        # img=cv2.resize(np.dstack((blue,img,red)),(shape[1],int(shape[0]/10)),fx=0,fy=0)


        cv2.imshow('small',img)
        # cv2.imshow('small',np.dstack((blue,img,red)))




        # for yy,line in enumerate(image):
        #     if np.count_nonzero(line)>10:
        #         x_val_hist=[]
        #         counter=0
        #         for x in line:
        #             if x>0:
        #                 x_val_hist.append(counter)
        #             counter=counter+1
        #         xmin=np.min(np.array(x_val_hist))
        #         xmax=np.max(np.array(x_val_hist))
        #         if np.abs(xmax-xmin)>300:
        #             center=(xmin+xmax)/2.0
        #             left=[(x,yy) for x in x_val_hist if x<center]
        #             right=[(x,yy) for x in x_val_hist if x>=center]
        #             for l in left:
        #                 points[0].append(l)
        #             for r in right:
        #                 points[1].append(r)
        return points

    def plotFitted(self,yvals,leftx,rightx):
        # # Generate some fake data to represent lane-line pixels
        # yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
        # leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        #                               for idx, elem in enumerate(yvals)])
        # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        # rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
        #                                 for idx, elem in enumerate(yvals)])
        # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
        # self.left_fit = np.polyfit(yvals, leftx, 2)
        # self.left_fitx = self.left_fit[0]*yvals**2 + self.left_fit[1]*yvals + self.left_fit[2]
        # self.right_fit = np.polyfit(yvals, rightx, 2)
        # self.right_fitx = self.right_fit[0]*yvals**2 + self.right_fit[1]*yvals + self.right_fit[2]
        # Plot up the fake data
        # plt.plot(leftx, yvals, 'o', color='red')
        # plt.plot(rightx, yvals, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(self.left_fitx, yvals, color='green', linewidth=3)
        plt.plot(self.right_fitx, yvals, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()
    def processVideo(self,fname):
        cap = cv2.VideoCapture(fname)

        self.old_l_x=[]
        self.old_l_y=[]
        self.old_r_x=[]
        self.old_r_y=[]

        if cap.isOpened():
            ret, img = cap.read()
            if not img is None:
                res=self.processSingleImage(img)
                cv2.imshow('Result',res)
                cv2.waitKey(1)
        while(cap.isOpened()):

            for ttt in range(1):
                ret, img = cap.read()

            if img is None:
                break

            res=self.processNextImage(img)
            cv2.imshow('Result',res)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

    def undistort(self,img):
        return cv2.undistort(img,calib['matrix'], calib['dist'], None,None)

    def unwarp(self,undist):
        src = np.float32(self.roi_corners)

        # pts = np.array(src, np.int32)
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(undist,[pts],True,(0,0,255))

        offset=100
        warped_size=(300+2*offset,300)
        dst = np.float32([
                            [offset, warped_size[1]],
                            [offset, 0],
                            [offset+warped_size[1], 0],
                            [offset+warped_size[1], warped_size[1]]])

        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        warped_orig = cv2.warpPerspective(undist, self.Mpersp, dsize=warped_size)
        return self.mask_lane_lines(warped_orig, s_thresh=(50,90))

    def removeNoise(self,img):
        kernel = np.ones((3,3),np.uint8)
        res = cv2.erode(img,kernel,iterations = 2)
        return cv2.dilate(res,kernel,iterations = 2)

    def removeTopBottom(self,img):
        res=img.copy()
        #remove car
        res[-int(res.shape[0]*0.1):-1,:,:]=0
        #remove sky
        res[0:int(self.roi_corners[1][1]),:,:]=0
        return res

    def processNextImage(self,img):
        undist=self.undistort(img)
        undist_masked=self.removeTopBottom(undist)
        warped=self.unwarp(undist_masked)
        # warped=self.removeNoise(warped)
        points=self.findLinePointsFast(warped)

        if len(points[0]) == 0 or len(points[1])==0:
            return img

        leftx,lefty= zip(*points[0])
        rightx,righty= zip(*points[1])

        self.all_y=np.array(list(range(warped.shape[0])))

        tl_x=list(leftx)+self.old_l_x
        tl_y=list(lefty)+self.old_l_y
        tr_x=list(rightx)+self.old_r_x
        tr_y=list(righty)+self.old_r_y


        # tl_x=list(leftx)
        # tl_y=list(lefty)
        # tr_x=list(rightx)
        # tr_y=list(righty)

        self.old_l_x=list(leftx)
        self.old_l_y=list(lefty)
        self.old_r_x=list(rightx)
        self.old_r_y=list(righty)

        # leftx=np.array(leftx)
        # lefty=np.array(lefty)
        # rightx=np.array(rightx)
        # righty=np.array(righty)

        tl_x=np.array(tl_x)
        tl_y=np.array(tl_y)
        tr_x=np.array(tr_x)
        tr_y=np.array(tr_y)
        # Fit a second order polynomial to each fake lane line
        self.left_fit = np.array(np.polyfit(tl_y, tl_x, 2))
        self.left_fitx = np.array(self.left_fit[0]*self.all_y**2 + self.left_fit[1]*self.all_y + self.left_fit[2])
        self.right_fit = np.array(np.polyfit(tr_y, tr_x, 2))
        self.right_fitx = np.array(self.right_fit[0]*self.all_y**2 + self.right_fit[1]*self.all_y + self.right_fit[2])

        # self.plotFitted(self.all_y,leftx,rightx)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.all_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.all_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv=np.linalg.inv(self.Mpersp)
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
        y_eval = np.max(self.all_y)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) \
                                     /np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) \
                                        /np.absolute(2*self.right_fit[0])
        # print(left_curverad, right_curverad)

        result=cv2.putText(result,'Curvature left: %.1f '%(left_curverad),(50,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
        result=cv2.putText(result,'Curvature right: %.1f '%(right_curverad),(50,100), self.font, 1,(255,255,255),2,cv2.LINE_AA)


        return result

    def processSingleImage(self,img):
        print(img.shape)
        undist=self.undistort(img)
        self.roi_corners=[[0.16*undist.shape[1],undist.shape[0]],
                    [0.45*undist.shape[1],0.63*undist.shape[0]],
                    [0.55*undist.shape[1],0.63*undist.shape[0]],
                    [0.84*undist.shape[1],undist.shape[0]]]
        undist_masked=self.removeTopBottom(undist)
        #Save before and after undistortion
        if not self.is_distortion_saved:
            dist_before_after=np.concatenate((img,undist), axis=1)
            dist_before_after=cv2.putText(dist_before_after,'Distorted',(50,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
            dist_before_after=cv2.putText(dist_before_after,'Undistorted',(50+img.shape[1],50), self.font, 1,(255,255,255),2,cv2.LINE_AA)

            cv2.imwrite('images/distortion.jpg',dist_before_after)
            self.is_distortion_saved=True

        warped=self.unwarp(undist_masked)
        # warped=self.removeNoise(warped)
        points=self.findLinePoints(warped)
        print(points[0])
        print(points[1])
        cv2.imshow('warped',warped)


        if len(points[0]) == 0 or len(points[1])==0:
            return img

        leftx,lefty= zip(*points[0])
        rightx,righty= zip(*points[1])

        self.all_y=np.array(list(range(warped.shape[0])))

        # tl_x=list(leftx)+self.old_l_x
        # tl_y=list(lefty)+self.old_l_y
        # tr_x=list(rightx)+self.old_r_x
        # tr_y=list(righty)+self.old_r_y


        tl_x=list(leftx)
        tl_y=list(lefty)
        tr_x=list(rightx)
        tr_y=list(righty)

        self.old_l_x=list(leftx)
        self.old_l_y=list(lefty)
        self.old_r_x=list(rightx)
        self.old_r_y=list(righty)

        # leftx=np.array(leftx)
        # lefty=np.array(lefty)
        # rightx=np.array(rightx)
        # righty=np.array(righty)

        tl_x=np.array(tl_x)
        tl_y=np.array(tl_y)
        tr_x=np.array(tr_x)
        tr_y=np.array(tr_y)
        # Fit a second order polynomial to each fake lane line
        self.left_fit = np.array(np.polyfit(tl_y, tl_x, 2))
        self.left_fitx = np.array(self.left_fit[0]*self.all_y**2 + self.left_fit[1]*self.all_y + self.left_fit[2])
        self.right_fit = np.array(np.polyfit(tr_y, tr_x, 2))
        self.right_fitx = np.array(self.right_fit[0]*self.all_y**2 + self.right_fit[1]*self.all_y + self.right_fit[2])

        # self.plotFitted(self.all_y,leftx,rightx)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.all_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.all_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv=np.linalg.inv(self.Mpersp)
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
        y_eval = np.max(self.all_y)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) \
                                     /np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) \
                                        /np.absolute(2*self.right_fit[0])
        print(left_curverad, right_curverad)

        result=cv2.putText(result,'Curvature left: %.1f '%(left_curverad),(50,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
        result=cv2.putText(result,'Curvature right: %.1f '%(right_curverad),(50,100), self.font, 1,(255,255,255),2,cv2.LINE_AA)


        return result


def main():
    det=LaneDetector()
    for fimg in images:
        img=cv2.imread(fimg)
        res=det.processSingleImage(img)
        cv2.imshow('result',res)
        cv2.waitKey(0)


    det.processVideo('CarND-Advanced-Lane-Lines/project_video.mp4')
    det.processVideo('CarND-Advanced-Lane-Lines/challenge_video.mp4')
    det.processVideo('CarND-Advanced-Lane-Lines/harder_challenge_video.mp4')

if __name__ == '__main__':
    main()
