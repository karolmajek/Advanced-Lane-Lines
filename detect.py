#!/usr/bin/python3
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import json

#Load calibration
calib=dict()
with open('calib.json', 'r') as f:
    calib=json.load(f)
calib['matrix']=np.array(calib['matrix'])
calib['dist']=np.array(calib['dist'])

class LaneDetector:
    """
    My class for lane detetion.
    """
    is_distortion_saved=False
    font = cv2.FONT_HERSHEY_SIMPLEX

    def mask_lane_lines(self,img):
        '''
        Method masks lane lines.
        '''
        img = np.copy(img)

        #Blur
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)

        #YUV for histogram equalization
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        img_wht = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        #Compute white mask
        img_ylw = cv2.cvtColor(img_wht, cv2.COLOR_BGR2HSV)
        img_wht=img_wht[:,:,1]
        img_wht[img_wht<250]=0
        mask_wht = cv2.inRange(img_wht, 250, 255)

        #Compute yellow mask
        lower_ylw = np.array([60,135,50])
        upper_ylw = np.array([180,155,155])
        mask_ylw = cv2.inRange(yuv, lower_ylw, upper_ylw)

        #Merge mask results
        mask = mask_wht+mask_ylw
        return mask

    def findLinePoints(self,image):
        '''
        Find lane lines points in a single image - slow.
        '''
        points=([],[])
        shape=image.shape
        img=image.copy()
        #Prepare images for visualization
        red=np.zeros_like(img)
        green=np.zeros_like(img)
        blue=np.zeros_like(img)

        #Set center to width/2
        center=int(shape[1]/2)

        #For each row starting from bottom
        for yy,line in list(enumerate(image))[::-1]:
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            if len(x_val_hist)>0:
                cv2.circle(green,(int(center),yy),1,(255,255,255))

                #Split to left/right line
                left=[(x,yy) for x in x_val_hist if x<center]
                right=[(x,yy) for x in x_val_hist if x>=center]

                if len(left)>0:
                    #Compute average
                    l=np.mean(np.array(left),axis=0)
                    l=(l[0],l[1])
                    center=l[0]+int(shape[1]*0.2)
                    cv2.circle(red,(int(l[0]),int(l[1])),1,(255,255,255))
                    #Add to points
                    points[0].append(l)
                if len(right)>0:
                    #Compute average
                    r=np.mean(np.array(right),axis=0)
                    r=(r[0],r[1])
                    cv2.circle(blue,(int(r[0]),int(r[1])),1,(255,255,255))
                    #Add to points
                    points[1].append(r)
        if False: #for debug
            img=cv2.resize(np.dstack((blue,green,red)),(shape[1],int(shape[0])),fx=0,fy=0)
            cv2.imshow('lines',img)
        return points

    def findLinePointsFast(self,image):
        '''
        Find lane lines points in video frame - fast.
        Works starting from second frame.
        '''
        points=([],[])
        shape=image.shape
        #Prepare images for visualization

        img=image.copy()
        red=np.zeros_like(img)
        blue=np.zeros_like(img)

        #For every 10th row starting from bottom
        for y,lx,rx,line in list(zip(self.all_y,self.left_fitx,self.right_fitx,image))[::-10]:
            lxmin=int(lx-20-0.2*(img.shape[0]-y))
            lxmax=int(lx+20+0.2*(img.shape[0]-y))
            rxmin=int(rx-20-0.2*(img.shape[0]-y))
            rxmax=int(rx+20+0.2*(img.shape[0]-y))
            cv2.circle(red,(lxmin,int(y)),1,(255,255,255))
            cv2.circle(red,(lxmax,int(y)),1,(255,255,255))
            cv2.circle(red,(rxmin,int(y)),1,(255,255,255))
            cv2.circle(red,(rxmax,int(y)),1,(255,255,255))
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            if len(x_val_hist)>5:
                #split points to left/right
                left=[(x,y) for x in x_val_hist if x<=lxmax and x>=lxmin]
                right=[(x,y) for x in x_val_hist if x>=rxmin and x<=rxmax]
                l=None
                r=None
                #Compute means for left/right
                if len(left):
                    l=np.mean(np.array(left),axis=0)
                if len(right):
                    r=np.mean(np.array(right),axis=0)
                if l is None or r is None or r[0]-l[0]>200:
                    if (not l is None) and l[0]>lxmin and l[0]<lxmax:
                        cv2.circle(blue,(int(l[0]),int(l[1])),1,(255,255,255))
                        points[0].append(l)
                    if (not r is None) and r[0]>rxmin and r[0]<rxmax:
                        cv2.circle(blue,(int(r[0]),int(r[1])),1,(255,255,255))
                        points[1].append(r)


        #Show roi for video frame
        img=cv2.resize(np.dstack((blue,red,red)),(shape[1],int(shape[0])),fx=0,fy=0)
        cv2.imshow('lines-video',img)
        return points

    def processVideo(self,fname):
        '''
        Process video data from file and save result to results dir
        '''
        self.old_l_x=None
        self.old_l_y=None
        self.old_r_x=None
        self.old_r_y=None
        #Open video
        cap = cv2.VideoCapture(fname)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        video_out = cv2.VideoWriter('results/'+fname.split('/')[-1],fourcc, 25, (1280,720))

        if cap.isOpened():
            ret, img = cap.read()
            #Process first image as single image
            if not img is None:
                res=self.processSingleImage(img)
                cv2.imshow('Result',res)
                cv2.waitKey(1)

        while(cap.isOpened()):
            ret, img = cap.read()
            if img is None:
                break
            res=self.processNextImage(img)
            # write result
            video_out.write(res)
            #Show result
            cv2.imshow('Result',res)
            cv2.waitKey(1)
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()

    def undistort(self,img):
        '''
        Undistort image
        '''
        return cv2.undistort(img,calib['matrix'], calib['dist'], None,None)

    def unwarp(self,undist):
        '''
        Unwarp image
        '''
        src = np.float32(self.roi_corners)

        offset=100
        warped_size=(300+2*offset,300)
        dst = np.float32([
                            [offset, warped_size[1]],
                            [offset, 0],
                            [offset+warped_size[1], 0],
                            [offset+warped_size[1], warped_size[1]]])

        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        warped_orig = cv2.warpPerspective(undist, self.Mpersp, dsize=warped_size)
        return warped_orig

    def removeTopBottom(self,img):
        '''
        Remove (set to black) top and bottom lines of image
        '''
        res=img.copy()
        #remove car
        res[-int(res.shape[0]*0.1):-1,:,:]=0
        #remove sky
        res[0:int(self.roi_corners[1][1]),:,:]=0
        return res

    def computeAndShow(self,img):
        '''
        Compute parameters:
        - left line curvature
        - right line curvature
        - distance from center

        And render result
        '''
        # Define y-value where we want radius of curvature
        y_eval = np.max(self.all_y)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.array(self.tl_y)*ym_per_pix, np.array(self.tl_x)*xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array(self.tr_y)*ym_per_pix, np.array(self.tr_x)*xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        #Distance from center
        img_center=img.shape[1]/2
        lane_center=self.left_fitx[-1]+self.right_fitx[-1]
        diff=lane_center-img_center
        self.diffm=diff*xm_per_pix

        img=cv2.putText(img,'Curvature left: %.1f m'%(self.left_curverad),(50,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
        img=cv2.putText(img,'Curvature right: %.1f m'%(self.right_curverad),(50,100), self.font, 1,(255,255,255),2,cv2.LINE_AA)
        img=cv2.putText(img,'Dist from center: %.1f m'%(self.diffm),(50,150), self.font, 1,(255,255,255),2,cv2.LINE_AA)

    def fitAndShow(self,warped,undist,points):
        '''
        Fit points and show result on image
        '''
        if len(points[0]) >= 5 and len(points[1])>=5:
            leftx,lefty= zip(*points[0])
            rightx,righty= zip(*points[1])

            self.all_y=np.array(list(range(warped.shape[0])))

            #Merge last points if not first frame
            if self.old_l_x is None or self.old_l_y is None or self.old_r_x is None or self.old_r_y is None:
                self.tl_x=list(leftx)
                self.tl_y=list(lefty)
                self.tr_x=list(rightx)
                self.tr_y=list(righty)
            else:
                self.tl_x=list(leftx)+self.old_l_x
                self.tl_y=list(lefty)+self.old_l_y
                self.tr_x=list(rightx)+self.old_r_x
                self.tr_y=list(righty)+self.old_r_y

            self.old_l_x=list(leftx)
            self.old_l_y=list(lefty)
            self.old_r_x=list(rightx)
            self.old_r_y=list(righty)

            #convert to numpy
            self.tl_x=np.array(self.tl_x)
            self.tl_y=np.array(self.tl_y)
            self.tr_x=np.array(self.tr_x)
            self.tr_y=np.array(self.tr_y)

            # Fit a second order polynomial to each fake lane line
            self.left_fit = np.array(np.polyfit(self.tl_y, self.tl_x, 2))
            self.left_fitx = np.array(self.left_fit[0]*self.all_y**2 + self.left_fit[1]*self.all_y + self.left_fit[2])
            self.right_fit = np.array(np.polyfit(self.tr_y, self.tr_x, 2))
            self.right_fitx = np.array(self.right_fit[0]*self.all_y**2 + self.right_fit[1]*self.all_y + self.right_fit[2])

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
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)

        unwarped = 255*cv2.warpPerspective(warped, Minv, (undist.shape[1], undist.shape[0])).astype(np.uint8)
        unwarped=np.dstack((np.zeros_like(unwarped),np.zeros_like(unwarped),unwarped))

        line_image = cv2.addWeighted(result, 1, unwarped, 1, 0)
        return result,line_image

    def processNextImage(self,img):
        '''
        Process image using information from previous one to speed up on video.
        '''
        undist=self.undistort(img)
        undist_masked=self.removeTopBottom(undist)
        warped=self.unwarp(undist_masked)
        warped=self.mask_lane_lines(warped)
        points=self.findLinePointsFast(warped)
        # cv2.imshow('warped',warped)
        result,_ = self.fitAndShow(warped,undist,points)
        self.computeAndShow(result)

        return result

    def processSingleImage(self,img):
        '''
        Process single image or first frame of video.
        '''
        self.old_l_x=None
        self.old_l_y=None
        self.old_r_x=None
        self.old_r_y=None

        undist=self.undistort(img)
        self.roi_corners=[[0.16*undist.shape[1],undist.shape[0]],
                    [0.45*undist.shape[1],0.63*undist.shape[0]],
                    [0.55*undist.shape[1],0.63*undist.shape[0]],
                    [0.84*undist.shape[1],undist.shape[0]]]

        show_roi=False
        if show_roi:
            src = np.float32(self.roi_corners)

            pts = np.array(src, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(undist,[pts],True,(0,0,255))
            cv2.imshow('roi',undist)

        #remove top and bottom (sky, car)
        undist_masked=self.removeTopBottom(undist)
        # cv2.imshow('undist_masked',undist_masked)

        #Save before and after undistortion
        if not self.is_distortion_saved:
            dist_before_after=np.concatenate((img,undist), axis=1)
            dist_before_after=cv2.putText(dist_before_after,'Distorted',(50,50), self.font, 1,(255,255,255),2,cv2.LINE_AA)
            dist_before_after=cv2.putText(dist_before_after,'Undistorted',(50+img.shape[1],50), self.font, 1,(255,255,255),2,cv2.LINE_AA)

            cv2.imwrite('images/distortion.jpg',dist_before_after)
            self.is_distortion_saved=True

        warped=self.unwarp(undist_masked)
        # cv2.imshow('transformed',warped)
        warped=self.mask_lane_lines(warped)
        # cv2.imshow('binary',warped)
        points=self.findLinePoints(warped)
        # cv2.imshow('warped',warped)

        if len(points[0]) == 0 or len(points[1])==0:
            return img

        result,_ = self.fitAndShow(warped,undist,points)
        self.computeAndShow(result)

        return result


def main():
    #Create LaneDetector object
    det=LaneDetector()

    #Process images
    for fimg in glob.glob('CarND-Advanced-Lane-Lines/test_images/*.jpg'):
        print(fimg.split('/')[-1])
        #Read image
        img=cv2.imread(fimg)
        #Do processing
        res=det.processSingleImage(img)
        #Write result
        cv2.imwrite('results/'+fimg.split('/')[-1],res)
        #Show result
        cv2.imshow('Result',res)
        cv2.waitKey(250)

    #Process all videos
    for fvideo in glob.glob('CarND-Advanced-Lane-Lines/*.mp4'):
        det.processVideo(fvideo)

if __name__ == '__main__':
    main()
