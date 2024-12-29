import cv2
import numpy as np

def create_mask(image_path, shape='rectangle', position=(50, 50), size=(100, 100)):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image")
        return
    mask = np.zeros(image.shape[:2], dtype="uint8")
    
    # Draw the specified shape on the mask
    if shape == 'rectangle':
        cv2.rectangle(mask, position, (position[0] + size[0], position[1] + size[1]), 255, -1)
    elif shape == 'circle':
        radius = size[0] // 2
        cv2.circle(mask, (position[0] + radius, position[1] + radius), radius, 255, -1)
    elif shape == 'ellipse':
        cv2.ellipse(mask, (position[0] + size[0] // 2, position[1] + size[1] // 2), (size[0] // 2, size[1] // 2), 0, 0, 360, 255, -1)
    else:
        print("Error: Unsupported shape")
        return
    # Create white mask image
    white_mask = np.ones_like(image) * 255
    white_masked_shape = cv2.bitwise_and(white_mask, white_mask, mask=mask)
    cv2.imwrite('masked_image.png', white_masked_shape)
    # Display the original image and white masked shape
    # cv2.imshow("Original Image", image)
    # cv2.imshow("White Masked Shape", white_masked_shape)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

create_mask('1.jpg', shape='rectangle', position=(100, 100), size=(200, 200))
