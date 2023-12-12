from PIL import Image
import numpy as np

# Example NumPy array representing an RGB image (random data for illustration)
# width = 100
# height = 100

# Create a random RGB array (values between 0 and 255 for each channel)
# img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

file = open("TestImages/GreyIshigami.ppm")
lines = file.readlines()
file.close()

lines = lines[2:]

rows, cols = map(int, lines[0].split(" "))

lines = lines[2:]

lines = list(map(int,lines))

img_array = np.array(lines).reshape((cols,rows))
print(img_array)

# Convert the NumPy array to a Pillow image
img_pil = Image.fromarray(img_array)

# Display the image (optional)
img_pil.show()

# # Save the image
img_pil.save("image.ppm")
print(img_pil)
