import cv2

# Read the image
img = cv2.imread("./pic/taiwan_center_satellite_combined.png")

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or failed to load. Check the file path.")
else:
    # Apply median blur
    img = cv2.medianBlur(img, 5)
    # img = cv2.medianBlur(img, 5)

    # Display the image
    cv2.imwrite("Blurred_Image.jpg", img)  # Provide a window name as the first argument
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows()
