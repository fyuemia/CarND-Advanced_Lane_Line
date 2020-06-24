import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg

# read in and make a list of calibration images
images = glob.glob("../camera_cal/calibration*.jpg")

# prepare object points
nx = 9
ny = 6

# arrays to store object points and image points from all the images
objPoints = []
imgPoints = []

# prepare objects points
objP = np.zeros((nx * ny, 3), np.float32)
objP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for i, fName in enumerate(images):
    # read in each image
    img = cv2.imread(fName)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret:
        imgPoints.append(corners)
        objPoints.append(objP)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imwrite(fName.replace("camera_cal", "camera_cal_output").replace("calibration", "draw_corners_calibration"), img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
camera_param = {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs":tvecs}
np.save("../param/camera_param.npy", camera_param)

for i, fName in enumerate(images):
    # read in each image
    img = cv2.imread(fName)

    # distortion correcting
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(fName.replace("camera_cal", "camera_cal_output"))
