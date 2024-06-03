import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image
image = cv2.imread('png/image/noise1.png')

# Apply Non-Local Means Denoising for noise reduction
denoised = cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)

# Contrast Stretching
p_min = np.min(denoised)
p_max = np.max(denoised)
stretched = np.zeros_like(denoised, dtype=np.uint8)

for i in range(denoised.shape[0]):
    for j in range(denoised.shape[1]):
        stretched[i, j] = (255 / (p_max - p_min)) * (denoised[i, j] - p_min)

# Convert to YCrCb color space
ycrcb = cv2.cvtColor(stretched, cv2.COLOR_BGR2YCrCb)

# Equalize the histogram of the Y channel
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

# Convert back to BGR color space
equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Resize the enhanced image to a medium size (e.g., 800x600)
medium_size = (800, 600)
resized_equalized = cv2.resize(equalized, medium_size, interpolation=cv2.INTER_AREA)

# Convert BGR image to RGB for displaying in Matplotlib
enhanced_image_rgb = cv2.cvtColor(resized_equalized, cv2.COLOR_BGR2RGB)

# Display the enhanced image using Matplotlib with a save option
plt.figure(figsize=(8, 6))
plt.imshow(enhanced_image_rgb)
plt.title('Enhanced Image')
plt.axis('off')

# Save the enhanced image button
save_button = plt.axes([0.8, 0.02, 0.1, 0.075])
button = plt.Button(save_button, 'Save Image')

def save_image(event):
    plt.imsave('enhanced_image.png', enhanced_image_rgb)
    print('Enhanced image saved as enhanced_image.png')

button.on_clicked(save_image)

plt.show()
