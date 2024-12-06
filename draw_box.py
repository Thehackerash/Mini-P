import cv2


def draw_box_on_image(image_path, coords, output_path):
    """
    Draws a rectangle on an image using the given coordinates and saves the updated image.

    Args:
        image_path (str): Path to the input image.
        coords (list): List of coordinates [x1, y1, x2, y2].
        output_path (str): Path to save the updated image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Extract coordinates
    x1, y1, x2, y2 = coords

    # Correct the rectangle coordinates
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Draw the rectangle on the image (blue color, thickness 2)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Save the updated image
    cv2.imwrite(output_path, image)
    print(f"Updated image saved to {output_path}")
