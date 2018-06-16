from ImageData import train_test_split,load_image
import matplotlib.pyplot as plt

images, labels, image_names, category =load_image(100)

data = train_test_split(image_size=128, test_size=0.3)

print("images in test set",len(data.train.images))

print('displaying a loaded image ')
plt.imshow(images[123])
plt.show()


