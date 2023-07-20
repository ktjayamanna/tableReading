from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.transform import hough_line, hough_line_peaks

image_path = "/home/kor/Documents/grays.ai/table_detection/OCR_tablenet/data/sample_table.png"

pil_image = Image.open(image_path)
numpy_array = np.array(pil_image)

if len(numpy_array.shape) > 2:
    grayscale_array = np.dot(numpy_array[..., :3], [0.2989, 0.5870, 0.1140])
else:
    grayscale_array = numpy_array

sobel_x = sobel(grayscale_array, axis=0)
sobel_y = sobel(grayscale_array, axis=1)
combined_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

h, theta, d = hough_line(combined_edges)
lines = hough_line_peaks(h, theta, d)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
axes[0].imshow(numpy_array)
axes[0].axis('off')
axes[1].imshow(sobel_x, cmap='gray')
axes[1].axis('off')
axes[2].imshow(sobel_y, cmap='gray')
axes[2].axis('off')
axes[3].imshow(combined_edges, cmap='gray')
axes[3].axis('off')

plt.tight_layout()
plt.show()
