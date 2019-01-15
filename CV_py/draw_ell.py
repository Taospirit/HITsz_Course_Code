import cv2
import numpy as np

img = np.zeros([700, 1000, 3], np.uint8)
cv2.rectangle(img, (0, 0), (999, 699), (255, 255, 255), -1)

cv2.ellipse(img, (150, 80), (100, 50), 0, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (350, 80), (100, 50), 10, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (550, 80), (100, 50), 20, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (720, 80), (100, 50), 30, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (900, 80), (100, 50), 40, 0, 360, (0, 0, 0), -1)


cv2.ellipse(img, (100, 250), (100, 50), 50, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (220, 250), (100, 50), 60, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (340, 250), (100, 50), 70, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (460, 250), (100, 50), 80, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (580, 250), (100, 50), 90, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (700, 250), (100, 50), 100, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (820, 250), (100, 50), 110, 0, 360, (0, 0, 0), -1)


cv2.ellipse(img, (100, 450), (100, 50), 120, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (220, 450), (100, 50), 130, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (360, 450), (100, 50), 140, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (530, 450), (100, 50), 150, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (700, 450), (100, 50), 160, 0, 360, (0, 0, 0), -1)

cv2.ellipse(img, (900, 450), (100, 50), 170, 0, 360, (0, 0, 0), -1)



# cv2.ellipse(img, (500,300), (13, 10), 45, 0, 360, (0, 0, 0), -1)
#
# cv2.ellipse(img, (600,300), (10, 8), 45, 0, 360, (0, 0, 0), -1)
#
# cv2.ellipse(img, (700,300), (8, 3), 45, 0, 360, (0, 0, 0), -1)
#
# cv2.ellipse(img, (800,300), (5, 6), 45, 0, 360, (0, 0, 0), -1)


cv2.imshow('ell',img)
cv2.imwrite('D:\img\ell.png', img)

k = cv2.waitKey(0) & 0XFF
if k == 27:
    cv2.destroyAllWindows()
