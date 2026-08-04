import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt


##### Read Image
img = cv2.imread(r"C:\Users\sudip\Downloads\vk\670407946_18583084804063583_477444808172154581_n.jpg")
print("Image:\n", img)
print("\n Image Shape:\n", img.shape)


'''Computer Vision (cv2) and Matplotlib (plt)'''

##### Display Image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img)
plt.title("\n Plot Image\n")
plt.axis('off')
plt.show()


##### Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

plt.imshow(img, cmap='gray')
plt.title("\n GrayScale Img \n")
plt.axis('off')
plt.show()


##### BGR - RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.title("\n BGR - RGB \n")
plt.imshow(rgb_img)
plt.axis('off')
plt.show()


#### Resize image
resize = cv2.resize(img, (800, 510))
cv2.imshow("Resize", resize)
cv2.waitKey(0)


#### Gaussian Blur
blur = cv2.GaussianBlur(img, (7,7), 5)
cv2.imshow("Blur", blur)
cv2.waitKey(0)

blur = cv2.GaussianBlur(rgb_img,(7,7),5)
plt.title("\n Gaussian Blur \n")
plt.imshow(blur)
plt.axis('off')
plt.show()


##### Edge Detection
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

edges = cv2.Canny(img, 50, 200)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.show()