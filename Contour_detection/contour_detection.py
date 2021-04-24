import cv2
import numpy as np

image = cv2.imread("countor_test.jpeg")
cv2.imwrite('rgb_image1.jpg', image)

# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image1.jpg', img_gray)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

img2 = cv2.merge((thresh,thresh,thresh))

cv2.imwrite('image_thres1.jpg', thresh)


# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

cv2.imwrite('contours__image1.jpg', image_copy)

final_img=np.hstack((image,img2,image_copy))

cv2.imwrite("final_image.jpg",final_img)
cv2.destroyAllWindows()


