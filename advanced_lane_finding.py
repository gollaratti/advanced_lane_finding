# -*- coding: utf-8 -*-
"""

v2.0 - 23-Feb-2017
Changes:
(1) In perspective_transformation(): Corrected persective transformation source and destination points.
(2) In edge_detect(): Corrected color conversion.
(3) In  detect_lanes() and  opt_detect_lanes(): Corrected calculation of radii of curvature and vehicle offset.
(4) In class Line(): set the lane curve fit averaging to 3.
(5) In process_video_frame(image): added color conversion and improved the logic for calling etect_lanes() and  opt_detect_lanes()


Created on Wed Feb 15 18:30:51 2017

Advanced Lane Finding v1.0

"""
##############################################################################
### INCLUDES
##############################################################################
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

##############################################################################
## Camera Calibration
##############################################################################
def calibrate_camera(list_images, num_corners = (6,9)):
    
    row, col = num_corners[0], num_corners[1]
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane.

    # Step through the list and search for chessboard corners
    for fname in list_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (col,row),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (col,row), corners, ret)


    # Calibrate the camera with the found object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


# Make a list of calibration images
list_images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist = calibrate_camera(list_images,(6,9))


if 1:
    img = cv2.imread('./test_images/test3.jpg')
    cv2.imwrite('./output_images/test_image.jpg', img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('./output_images/undistorted_test_image.jpg', dst)
else :        
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('distorted image')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    ax2.set_title('undistorted image')
    ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))


##############################################################################
# Functions for absolute value of gradient along x orientation, magnitude of 
#the gradients and direction of the gradient
##############################################################################
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    abs_sobel_binary = np.zeros_like(scaled_sobel)
    abs_sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 5) Return this mask   
    return abs_sobel_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Take the derivative in x and y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    # 2) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # 3) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 4) Return this mask  
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize = sobel_kernel)
    # 2) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradients = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradients)
    dir_binary[(gradients >=thresh[0]) & (gradients <= thresh[1])] = 1   
    return dir_binary

##############################################################################
##Prepare image for lane detection using magnitude threshold and S component 
## of HLS converted image.
##############################################################################
def edge_detect(undist_image):
    gray = cv2.cvtColor(undist_image, cv2.COLOR_BGR2GRAY)
    mag_binary = mag_thresh(gray, sobel_kernel=9, mag_thresh=(75, 255))
    hls = cv2.cvtColor(undist_image, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    
    thresh = (175, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    comb_binary = np.zeros_like(binary)
    comb_binary[(mag_binary == 1) | (binary >= 1)] = 1
    
    return comb_binary

comb_binary = edge_detect(dst)

if 0:
    cv2.imwrite('./output_images/combined_binary_test_image.jpg', comb_binary)
else:
    f2, (a2) = plt.subplots(1, 1, figsize=(20,10))
#    a1.set_title('Test image - undistorted')
#    a1.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    a2.set_title('Lane detected with window search')
    a2.imshow(comb_binary, cmap='gray')

##############################################################################
##Perspective transformation
##############################################################################

def perspective_transformation():
    #define source image points 
    src = np.array([[215,700], [1080,700], [735,480],[550,480]], np.int32)
    #define destination image points for bird's eye view
    dst = np.array([[360,720], [960,720], [960,0], [360,0 ]], np.int32)  
    
    M = cv2.getPerspectiveTransform(np.float32(src),np.float32(dst))
    Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    
    return M, Minv

M,Minv = perspective_transformation()
binary_warped = cv2.warpPerspective(comb_binary,M,(comb_binary.shape[1], comb_binary.shape[0]), flags=cv2.INTER_LINEAR)

if 0:
    cv2.imwrite('./output_images/binary_warped_test_image.jpg', binary_warped)
#else:
#    f2, (a2) = plt.subplots(1, 1, figsize=(20,10))
##    a1.set_title(' edge detected image')
##    a1.imshow(comb_binary, cmap='gray')
#
#    a2.set_title('Lane detected perspective transformed image')
#    a2.imshow(binary_warped, cmap='gray')



##############################################################################
##Detect lanes - using histogram and window search method
##############################################################################
def detect_lanes(binary_warped, visualize = True):
    # Histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    if (visualize == True):
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if (visualize == True):        
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # If we don't find enough relevant points, return all None, this would trigger 
    # using previous frame data for videos
    min_inds = 7200
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None, None, None, None, None
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    if (visualize == True):             
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Show and save this to image on disk
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.imsave('./output_images/window_search_lane_detect_test_image.jpg', out_img)
        
    ## Radius of curvature
    # Define y-value where we want radius of curvature
    y_eval = 600
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 60/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])   
    
    # Calculate vehicle offset from lane center
    bottom_y = binary_warped.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = binary_warped.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    vehicle_offset *= xm_per_pix
    
   
    return left_fit, right_fit, left_curverad, right_curverad, vehicle_offset

left_fit, right_fit, left_curverad, right_curverad, vehicle_offset = detect_lanes(binary_warped, visualize=True)


##############################################################################
##Optimised Detect lanes - this is used for subsequent frames on videos
##############################################################################
def opt_detect_lanes(binary_warped, left_fit, right_fit, visualize=True):
    # Binary warped image from the  next frame of video 
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # If we don't find enough relevant points, return all None. This triggers 
    # detection of lanes using histogram and window method. 
    min_inds = 7200
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None, None, None, None, None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if (visualize == True):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # show and save image to disk
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.imsave('./output_images/quick_search_lane_detect_test_image.jpg')
    
     ## Radius of curvature
    # Define y-value where we want radius of curvature
    y_eval = 600
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 60/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])  
    
    # Calculate vehicle offset from lane center
    bottom_y = binary_warped.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = binary_warped.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    vehicle_offset *= xm_per_pix
    
    
    return left_fit, right_fit, left_curverad, right_curverad, vehicle_offset


##############################################################################
##Draw lanes on original image for visualization
##############################################################################
def draw_lanes_on_road(binary_warped, left_fit, right_fit, rc, vehicle_offset, image ):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image,1, newwarp, 0.3, 0)
       
    # Print the radius of curvature and vehicle offset
    label = 'Radius of curvature: %.f m' % rc
    result = cv2.putText(result, label, (20,50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
    label = 'Car offset from lane center: %.1f m' % vehicle_offset
    result = cv2.putText(result, label, (20,80), 0, 1, (255,255,255), 2, cv2.LINE_AA)

    
    return result

result = draw_lanes_on_road(binary_warped, left_fit, right_fit, (left_curverad + right_curverad)/2.0, vehicle_offset, dst)    
    
if 0:
    cv2.imwrite('./output_images/final_test_image.jpg', result)

##############################################################################
##Line class for videos
##############################################################################    
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # Polynomial coefficients: x = A*y^2 + B*y + C
        self.A = []
        self.B = []
        self.C = []
        
        # Moving average of co-efficients
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.
        
        # radius of curvature
        self.rc = 0.
        #car offset from center
        self.vehicle_offset = 0.

    def get_average_fit(self):
        return(self.A_avg, self.B_avg, self.C_avg)
    def average_fit(self, fit_coeffs):
                              
        self.A.append(fit_coeffs[0])
        self.B.append(fit_coeffs[1])
        self.C.append(fit_coeffs[2])
        
        # pop out the oldest co-efficients and average them over 3 frames
        if(len(self.A) >=2):
            _ = self.A.pop(0)
            _ = self.B.pop(0)
            _ = self.C.pop(0)
                    
        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)
               
        return self.A_avg, self.B_avg, self.C_avg
    def set_params(self, rc, vehicle_offset):
        self.rc = rc
        self.vehicle_offset = vehicle_offset
    def get_params(self):
        return self.rc, self.vehicle_offset
  
##############################################################################
##Pipeline on Video frames
############################################################################## 

# define global variables
left_lane = Line()
right_lane = Line()
lane_detect = False
new_fit, reuse_fit = 0, 0


def process_video_frame(image):
    
    global mtx, dist, left_lane, right_lane, lane_detect, new_fit, reuse_fit
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #undistort the image
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    
    #edge detected imag
    comb_binary = edge_detect(dst)
    
    #perspective transformation
    binary_warped = cv2.warpPerspective(comb_binary,M,(comb_binary.shape[1], comb_binary.shape[0]), flags=cv2.INTER_LINEAR)
   
    if (lane_detect == False):             
        #detect lanes with window searching
        left_fit, right_fit, left_curverad, right_curverad, vehicle_offset = detect_lanes(binary_warped, visualize=False)
        
        if ((right_fit == None) or (left_fit == None)):
            left_fit = left_lane.get_average_fit()
            right_fit = right_lane.get_average_fit()
            rc, vehicle_offset = left_lane.get_params()
            reuse_fit+=1
        else:
            left_fit = left_lane.average_fit(left_fit)
            right_fit = right_lane.average_fit(right_fit)
            #radius of curvature and vehicle offset
            rc = (left_curverad  + right_curverad)/2.0
            left_lane.set_params(rc,vehicle_offset)
            lane_detect = True
            new_fit+=1
    else:
        # fast lane detect
        left_fit = left_lane.get_average_fit()
        right_fit = right_lane.get_average_fit()
        
        left_fit, right_fit, left_curverad, right_curverad, vehicle_offset = opt_detect_lanes(binary_warped, left_fit, right_fit, visualize=False)
        if ((right_fit == None) or (left_fit == None)):
            #detect lanes with window searching
            left_fit, right_fit, left_curverad, right_curverad, vehicle_offset = detect_lanes(binary_warped, visualize=False)
            if ((right_fit == None) or (left_fit == None)):
                left_fit = left_lane.get_average_fit()
                right_fit = right_lane.get_average_fit()
                rc, vehicle_offset = left_lane.get_params()
                reuse_fit+=1
            else:
                left_fit = left_lane.average_fit(left_fit)
                right_fit = right_lane.average_fit(right_fit)
                #radius of curvature and vehicle offset
                rc = (left_curverad  + right_curverad)/2.0
                left_lane.set_params(rc,vehicle_offset)
                lane_detect = True
                new_fit+=1
        else:
            rc = (left_curverad  + right_curverad)/2.0
            reuse_fit+=1

    
    #draw lanes on the image
    result = draw_lanes_on_road(binary_warped, left_fit, right_fit, rc, vehicle_offset, dst)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
   
    return result

output_video = 'p4_project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_video_frame)
output_clip.write_videofile(output_video, audio=False)

print('Number of times histogram and window search is used:',new_fit)
print('Number of times quick lanes detect is used :', reuse_fit)
