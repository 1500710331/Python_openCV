import cv2 as cv
import numpy as nu
import pytesseract as tess
from PIL import Image


#对图片进行识别
def recognize_text(path):
	#灰度处理，二值化处理
	src = cv.imread(path)
	gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
	ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
	open_out = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
	

	#cv.imshow("binary_image", open_out)
	#cv.waitKey(0)
	#cv.destroyAllWindows()
	#进行识别
	cv.bitwise_not(open_out, open_out)
	textImage = Image.fromarray(open_out)
	text = tess.image_to_string(textImage)
	return text


'''
if __name__=="__main__":

	result = recognize_text('/home/st123456/Desktop/1.jpg')
	print(result)

print("-------------Python OpenCV Tutorial------------")
src = cv.imread('/home/st123456/Desktop/captcha.png')
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
recognize_text()
cv.waitKey(0)
cv.destroyAllWindows()
'''