import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# Read in an image and grayscale it
images = glob.glob("../test_images/*.jpg")

camera_param = np.load("../camera_cal_output/camera_param.npy", allow_pickle=True)[()]
mtx = camera_param["mtx"]
dist = camera_param["dist"]


def get_perspective_transform(img):
    # src_left_bottom = [150, 720]  # left bottom
    # src_right_bottom = [1250, 720]  # right bottom
    # src_left_top = [590, 450]  # left top
    # src_right_top = [700, 450]  # right top
    # image size
    img_size = (img.shape[1], img.shape[0])
    # src = np.float32([src_left_bottom, src_right_bottom, src_right_top, src_left_top])
    # offset = 200
    # dst = np.float32(
    #     [[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 63, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 20), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    return cv2.getPerspectiveTransform(src, dst)


def undistort_image(img, mtx=mtx, dist=dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'

    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max

    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, thresh_s=(0, 255), thresh_l=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    binary_output_s = np.zeros_like(s_channel)
    binary_output_s[(s_channel > thresh_s[0]) & (s_channel <= thresh_s[1])] = 1

    binary_output_l = np.zeros_like(l_channel)
    binary_output_l[(l_channel > thresh_l[0]) & (l_channel <= thresh_l[1])] = 1
    # 3) Return a binary image of threshold result
    binary_output = cv2.bitwise_and(binary_output_l, binary_output_s)
    return binary_output


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial_and_draw(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    points = np.vstack((left_fitx,ploty)).T.reshape(-1, 2)
    for index, item in enumerate(points):
        if index == ploty.shape[0] - 1:
            break
        cv2.line(out_img, tuple(item.astype(int)), tuple(points[index + 1].astype(int)), color=[0,255,0], thickness=10)

    points = np.vstack((right_fitx,ploty)).T.reshape(-1, 2)
    for index, item in enumerate(points):
        if index == ploty.shape[0] - 1:
            break
        cv2.line(out_img, tuple(item.astype(int)), tuple(points[index + 1].astype(int)), color=[0,255,0], thickness=10)

    return out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def roi(img, vertices):
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(img)
    ignore_mask_color = 1
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped,left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plots the left and right polynomials on the lane lines
    points = np.vstack((left_fitx,ploty)).T.reshape(-1, 2)
    for index, item in enumerate(points):
        if index == ploty.shape[0] - 1:
            break
        cv2.line(result, tuple(item.astype(int)), tuple(points[index + 1].astype(int)), color=[0,255,0], thickness=10)

    points = np.vstack((right_fitx,ploty)).T.reshape(-1, 2)
    for index, item in enumerate(points):
        if index == ploty.shape[0] - 1:
            break
        cv2.line(result, tuple(item.astype(int)), tuple(points[index + 1].astype(int)), color=[0,255,0], thickness=10)

    return result


# Unwarp Image and plot line

def DrawLine(original_image, binary_warped, Minv, left_fit, right_fit):
    h, w = binary_warped.shape
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, h - 1, num=h)  # to cover same y-range as image

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # axes[index+1].imshow(newwarp)
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)
    return result


def measure_curvature_pixels(ploty, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad


def measure_curv(leftx, lefty, rightx, righty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    '''
    Calcualtes the curvature of polynomial functions in meters and the offset from the lancenter
    '''
    Y_MAX = 720  # this is the image height
    X_MAX = 1280  # this is the image width
    # Transform pixel to meters
    leftx = leftx * xm_per_pix
    lefty = lefty * ym_per_pix
    rightx = rightx * xm_per_pix
    righty = righty * ym_per_pix

    # fit the polynomial
    left_fit_cr = np.polyfit(lefty, leftx, 2)
    right_fit_cr = np.polyfit(righty, rightx, 2)

    # Define y-value where we want radius of curvature
    # choose the maximum y-value
    y_eval = Y_MAX * ym_per_pix

    # Implement the caculation of R_curve
    # Caluate the radius R = (1+(2Ay+B)^2)^3/2 / (|2A|)
    radius_fun = lambda A, B, y: (1 + (2 * A * y + B) ** 2) ** (3 / 2) / abs(2 * A)

    left_curverad = radius_fun(left_fit_cr[0], left_fit_cr[1], y_eval)
    right_curverad = radius_fun(right_fit_cr[0], right_fit_cr[1], y_eval)

    return left_curverad, right_curverad


def measure_offset(leftx, lefty, rightx, righty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    '''
    calculate the the offest from lane center
    '''
    # HYPOTHESIS : the camera is mounted at the center of the car
    # the offset of the lane center from the center of the image is
    # distance from the center of lane

    Y_MAX = 720  # this is the image height
    X_MAX = 1280  # this is the image width
    # Transform pixel to meters
    leftx = leftx * xm_per_pix
    lefty = lefty * ym_per_pix
    rightx = rightx * xm_per_pix
    righty = righty * ym_per_pix

    # fit the polynomial
    left_fit_cr = np.polyfit(lefty, leftx, 2)
    right_fit_cr = np.polyfit(righty, rightx, 2)

    # Define y-value where we want radius of curvature
    # choose the maximum y-value
    y_eval = Y_MAX * ym_per_pix

    left_point = np.poly1d(left_fit_cr)(y_eval)
    right_point = np.poly1d(right_fit_cr)(y_eval)

    lane_center = (left_point + right_point) / 2
    image_center = X_MAX * xm_per_pix / 2

    offset = lane_center - image_center

    return offset


for fName in images:
    image = mpimg.imread(fName)
    img_size = (image.shape[1], image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 63, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 20), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    undist = undistort_image(image)

    hlsFiltered = hls_select(undist, thresh_s=(90, 255), thresh_l=(30, 255))

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 180))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(15, 160))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 160))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hlsFiltered == 1)] = 1

    # region of interest
    imshape = image.shape
    left_bottom = (120,imshape[0])
    left_top = (int(imshape[1]*0.45), int(imshape[0]*0.6))
    right_top = (int(imshape[1]*0.55), int(imshape[0]*0.6))
    right_bottom = (imshape[1]-50,imshape[0])
    vertices = np.array([[left_bottom,left_top, right_top, right_bottom]], dtype=np.int32)
    masked_combined = roi(combined,vertices)

    img_size = (masked_combined.shape[1], masked_combined.shape[0])
    warped = cv2.warpPerspective(masked_combined, M, img_size)

    laneImg = fit_polynomial_and_draw(warped)

    left_fit, right_fit = fit_polynomial(warped)
    lanePolyImg = search_around_poly(warped,left_fit,right_fit)

    result = DrawLine(image, warped,Minv,left_fit,right_fit)

    color = (0, 0, 255)
    linethickness = 3
    cv2.line(image, tuple(left_bottom), tuple(left_top), color, thickness=linethickness)
    cv2.line(image, tuple(left_top), tuple(right_top), color, thickness=linethickness)
    cv2.line(image, tuple(right_top), tuple(right_bottom), color, thickness=linethickness)
    cv2.line(image, tuple(right_bottom), tuple(left_bottom), color, thickness=linethickness)

    # imageList = {"origin": image,
    #              "combined": combined,
    #              "warped": warped,
    #              "lane": laneImg}
    print(image.shape)
    imageList = {"undist": undist,
                 "hls":hlsFiltered,
                 "gradx": gradx,
                 "grady": grady,
                 "mag": mag_binary,
                 "dir":dir_binary,
                 "combine":combined,
                 "masked_combine": masked_combined,
                 "warped":warped,
                 "lane":laneImg,
                 "lanePloy":lanePolyImg,
                 "result":result}

    # fig, axes = plt.subplots(nrows=(np.ceil(len(imageList)/2).astype(int)), ncols=2, figsize=(20, 40))
    # axes = axes.flatten()
    # for index, (ax, key) in enumerate(zip(axes, imageList.keys())):
    #     ax.imshow(imageList[key].astype(np.uint8), cmap="gray")
    #     ax.set_title(key, fontsize=40)
    #     ax.axis('off')
    # plt.suptitle(fName.split("/")[-1].replace(".jpg", ""),fontsize=50)
    # plt.savefig("../output_project_images/challenge/result_"+fName.split("/")[-1])
    # plt.close(fig)
    #
    for index, key in enumerate(imageList.keys()):
        fig = plt.figure()
        plt.imshow(imageList[key].astype(np.uint8), cmap="gray")
        plt.title(key)
        plt.savefig("../output_images/result_"+key+"_"+fName.split("/")[-1])
        plt.close(fig)