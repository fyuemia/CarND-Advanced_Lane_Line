import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg

camera_param = np.load("../camera_cal_output/camera_param.npy", allow_pickle=True)[()]
mtx = camera_param["mtx"]
dist = camera_param["dist"]

fName = "../test_images/straight_lines1.jpg"
img = mpimg.imread(fName)

imshape = img.shape

src_left_bottom=[150,720] #left bottom
src_right_bottom=[1250,720] #right bottom
src_left_top=[590,450] # left top
src_right_top=[700,450] # right top

def corners_unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # image size
    img_size = (gray.shape[1], gray.shape[0])
    # source points
    src = np.float32([src_left_bottom, src_right_bottom, src_right_top, src_left_top])
    # destination points
    offset = 200
    dst = np.float32([[offset, 0],[img_size[0] - offset, 0],[img_size[0] - offset, img_size[1]],[offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    # Return the resulting image and matrix
    return warped, M


if __name__=="__main__":
    images = glob.glob("../test_images/*.jpg")
    for fName in images:
        img = mpimg.imread(fName)

        warped, M = corners_unwarp(img, mtx, dist)

        color = (0, 0, 255)
        linethickness = 3
        cv2.line(img, tuple(src_left_bottom), tuple(src_left_top), color, thickness=linethickness)
        cv2.line(img, tuple(src_left_top), tuple(src_right_top), color, thickness=linethickness)
        cv2.line(img, tuple(src_right_top), tuple(src_right_bottom), color, thickness=linethickness)
        cv2.line(img, tuple(src_right_bottom), tuple(src_left_bottom), color, thickness=linethickness)
        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped)
        ax2.set_title('Perspective Transformed Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(fName.replace("test_images/", "output_images/perspective_"))
