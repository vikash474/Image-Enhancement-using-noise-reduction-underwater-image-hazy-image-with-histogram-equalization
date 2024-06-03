import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def enhance_underwater_image(image_path):
    # Load the underwater image
    image = cv2.imread(image_path)

    # Convert to float32 for gamma correction
    image_float = image.astype(np.float32) / 255.0

    # Gamma correction
    gamma = 1.5
    corrected = np.clip(image_float ** gamma, 0, 1.0)

    # Convert back to uint8
    corrected_uint8 = (corrected * 255).astype(np.uint8)

    # Convert to LAB color space for histogram equalization
    lab = cv2.cvtColor(corrected_uint8, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply histogram equalization to the L channel
    l_channel_eq = cv2.equalizeHist(l_channel)

    # Merge the channels back together
    lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return enhanced

def save_image(event):
    # Prompt the user to enter a file name and save the image
    save_path = input("Enter the path to save the image (e.g., enhanced_image.png): ").strip()
    plt.imsave(save_path, enhanced[:, :, ::-1])  # Convert BGR to RGB for saving
    print(f"Enhanced image saved to {save_path}")

if __name__ == "__main__":
    # Path to your underwater image
    image_path = 'png/water.png'
    enhanced = enhance_underwater_image(image_path)

    # Display the enhanced image using Matplotlib
    fig, ax = plt.subplots()
    plt.imshow(enhanced[:, :, ::-1])  # Convert BGR to RGB for displaying
    plt.axis('off')

    # Add a button to save the image
    save_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
    save_button = Button(save_ax, 'Save Enhanced Image', color='lightgoldenrodyellow', hovercolor='0.975')
    save_button.on_clicked(save_image)

    plt.show()
