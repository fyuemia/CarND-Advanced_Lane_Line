# the image process pipeline for the lane detection.

# import the libary
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# import the function
print("Import the functions used in pipeline...")
import src.helper as helper

# class to keep left/right line
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = []
        # distance in meters of vechicle center from the line
        self.line_base_pos = []
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []


class Pipeline():
    '''
    pipeline class with has left/right line class to hold the related information
    '''
    # Get cal and tranform parameter, set as class variable
    print("Import the camera calbration parameter & view_perspective tranform parameter...")
    mtx, dist = helper.get_camera_param()
    M, Minv = helper.get_transform_param()

    def __init__(self, left, right):
        '''
        initial with left, right line class
        '''
        self.left = left  # the Line() class to keep the left line info
        self.right = right  # the Line() class to keep the left line info
        self.image_counter = 0  # the image counter
        self.search_fail_counter = 0  # this is used to record lane pixel search failure.
        self.search_ok = False  # flag use to record lane search is ok or not
        self.smooth_number = 15  # use to average the radius valude to let the screen number not so jump
        self.debug_window = False  # Option if turn on the debug window in the pipeline
        self.radius = []  # store the radius data, arverage of left/right lane cur
        self.offset = []  # store the car center offset from lane center
        self.quick_search = False  # not implement, use last time fit line to quick search the lane points
        self.search_initialized = False  # not initialized use 1/slid window search, otherwise ploysearch,

    def project_debug_window(self, out_img):
        # set the debug window size
        fit_image_resize = cv2.resize(out_img, (640, 360))
        return fit_image_resize

    def store_search(self, leftx, lefty, rightx, righty):
        """
        update the search result
        """
        self.left.allx.append(leftx)
        self.left.ally.append(lefty)
        self.right.allx.append(rightx)
        self.right.ally.append(righty)

    def get_recent_search(self):
        """
        output recent search result
        """
        leftx = self.left.allx[-1]
        lefty = self.left.ally[-1]
        rightx = self.right.allx[-1]
        righty = self.right.ally[-1]

        return leftx, lefty, rightx, righty

    def store_fit(self, left_fitx, right_fitx):

        self.left.recent_xfitted.append(left_fitx)
        self.right.recent_xfitted.append(right_fitx)

    def get_recent_fit(self):
        left_fit = self.left.recent_xfitted[-1]
        right_fit = self.right.recent_xfitted[-1]

        return left_fit, right_fit

    def project_fit_lane_info(self, image, color=(0, 255, 255)):
        """
        project the fited lane information to the image
        use last 15 frame average data to avoid the number quick jump on screen.
        """
        offset = np.mean(self.offset[-15:-1]) if len(self.offset) > self.smooth_number else np.mean(self.offset)
        curverad = np.mean(self.radius[-15:-1]) if len(self.radius) > self.smooth_number else np.mean(self.radius)
        direction = "right" if offset < 0 else "left"
        str_cur = "Radius of Curvature = {}(m)".format(int(curverad))
        str_offset = "Vehicle is {0:.2f}m ".format(abs(offset)) + "{} of center".format(direction)
        cv2.putText(image, str_cur, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(image, str_offset, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    def pipeline(self, image):
        # counter the image
        self.image_counter += 1

        img_size = (image.shape[1], image.shape[0])

        # distort the image
        undist = helper.undistort_image(image,self.mtx,self.dist)

        # Choose a Sobel kernel size
        ksize = 3  # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        hlsFiltered = helper.hls_select(undist, thresh_s=(90, 255), thresh_l=(30, 255))
        gradx = helper.abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(10, 180))
        grady = helper.abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(15, 160))
        mag_binary = helper.mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(50, 160))
        dir_binary = helper.dir_threshold(undist, sobel_kernel=ksize, thresh=(0.7, 1.3))

        # combine different threshold
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hlsFiltered == 1)] = 1

        # apply region of interest for lane line search
        imshape = image.shape
        left_bottom = (120, imshape[0])
        left_top = (int(imshape[1] * 0.45), int(imshape[0] * 0.6))
        right_top = (int(imshape[1] * 0.55), int(imshape[0] * 0.6))
        right_bottom = (imshape[1] - 50, imshape[0])
        vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
        masked_combined = helper.roi(combined, vertices)

        # get warped image
        img_size = (masked_combined.shape[1], masked_combined.shape[0])
        warped = cv2.warpPerspective(masked_combined, self.M, img_size)

        # lane line search
        if not self.search_initialized:
            leftx, lefty, rightx, righty, out_img = helper.find_lane_pixels(warped)
            self.search_initialized = True
        else:
            left_fit, right_fit = self.get_recent_fit()
            leftx, lefty, rightx, righty, out_img = helper.search_around_poly(warped, left_fit,right_fit)


        # check the pixels search result, if the leftx or lefty is empyt, use recent data, if there is no recent data, return the image it self
        if leftx.size == 0 or rightx.size == 0:
            self.search_ok = False
            self.search_fail_counter += 1
            if self.left.allx == []:
                return image  # logical choise, only happend first frame search failed, or first n frame search failed
            else:  # use recent search result
                leftx, lefty, rightx, righty = self.get_recent_search()
        else:  # store the search result
            self.search_ok = True
            self.store_search(leftx, lefty, rightx, righty)

        # get the polynomial line points
        left_fit, right_fit, left_fitx, right_fitx, ploty = helper.fit_poly(img_size, leftx, lefty, rightx, righty)

        # store fit result
        self.store_fit(left_fit, right_fit)

        # measure the curverad and offset
        left_curverad, right_curverad = helper.measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30 / 720,
                                                     xm_per_pix=3.7 / 700)
        curverad = (left_curverad + right_curverad) / 2
        offset = helper.measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

        # store the lane data for furthre caculation
        self.radius.append(curverad)
        self.offset.append(offset)

        # draw result image
        result = helper.draw_lane_result(undist, warped, self.Minv, left_fit, right_fit)

        # write curverad and offset on to result image
        self.project_fit_lane_info(result, color=(0, 255, 255))

        if self.debug_window:
            # draw fitted line
            out_img = helper.draw_lane_fitted(out_img, left_fitx, right_fitx, ploty)
            debug_window = self.project_debug_window(out_img)
            result[:360, 640:] = debug_window

        return result


    def pipeline_challenge(self, image):
        # counter the image
        self.image_counter += 1

        img_size = (image.shape[1], image.shape[0])

        # distort the image
        undist = helper.undistort_image(image,self.mtx,self.dist)

        # Choose a Sobel kernel size
        ksize = 3  # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        hlsFiltered = helper.hls_select(undist, thresh_s=(90, 255), thresh_l=(30, 255))
        gradx = helper.abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(10, 180))
        grady = helper.abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(15, 160))
        mag_binary = helper.mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(50, 160))
        dir_binary = helper.dir_threshold(undist, sobel_kernel=ksize, thresh=(0.7, 1.3))

        # combine different threshold
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hlsFiltered == 1)] = 1

        # apply region of interest for lane line search
        imshape = image.shape
        left_bottom = (120, imshape[0])
        left_top = (int(imshape[1] * 0.45), int(imshape[0] * 0.6))
        right_top = (int(imshape[1] * 0.55), int(imshape[0] * 0.6))
        right_bottom = (imshape[1] - 50, imshape[0])
        vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
        masked_combined = helper.roi(combined, vertices)

        # get warped image
        img_size = (masked_combined.shape[1], masked_combined.shape[0])
        warped = cv2.warpPerspective(masked_combined, self.M, img_size)

        # lane line search
        if not self.search_initialized:
            leftx, lefty, rightx, righty, out_img = helper.find_lane_pixels(warped)
            self.search_initialized = True
        else:
            left_fit, right_fit = self.get_recent_fit()
            leftx, lefty, rightx, righty, out_img = helper.search_around_poly(warped, left_fit,right_fit)


        # check the pixels search result, if the leftx or lefty is empyt, use recent data, if there is no recent data, return the image it self
        if leftx.size == 0 or rightx.size == 0:
            self.search_ok = False
            self.search_fail_counter += 1
            if self.left.allx == []:
                return image  # logical choise, only happend first frame search failed, or first n frame search failed
            else:  # use recent search result
                leftx, lefty, rightx, righty = self.get_recent_search()
        else:  # store the search result
            self.search_ok = True
            self.store_search(leftx, lefty, rightx, righty)

        # get the polynomial line points
        left_fit, right_fit, left_fitx, right_fitx, ploty = helper.fit_poly(img_size, leftx, lefty, rightx, righty)

        # store fit result
        self.store_fit(left_fit, right_fit)

        # measure the curverad and offset
        left_curverad, right_curverad = helper.measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30 / 720,
                                                     xm_per_pix=3.7 / 700)
        curverad = (left_curverad + right_curverad) / 2
        offset = helper.measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

        # store the lane data for furthre caculation
        self.radius.append(curverad)
        self.offset.append(offset)

        # draw result image
        result = helper.draw_lane_result(undist, warped, self.Minv, left_fit, right_fit)

        # write curverad and offset on to result image
        self.project_fit_lane_info(result, color=(0, 255, 255))

        if self.debug_window:
            # draw fitted line
            out_img = helper.draw_lane_fitted(out_img, left_fitx, right_fitx, ploty)
            debug_window = self.project_debug_window(out_img)
            result[:360, 640:] = debug_window

        return result


