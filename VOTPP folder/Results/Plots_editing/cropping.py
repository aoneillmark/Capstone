from PIL import Image

def crop_image(input_image_path, output_image_path, crop_percentage):
    """
    Crop an image based on the given percentages.
    
    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the cropped image.
    :param crop_percentage: A tuple of four values (top, right, bottom, left) representing the crop percentage.
    """
    # Open the input image
    with Image.open(input_image_path) as img:
        width, height = img.size

        # Calculate crop dimensions
        left = width * crop_percentage[3] / 100
        top = height * crop_percentage[0] / 100
        right = width * (1 - crop_percentage[1] / 100)
        bottom = height * (1 - crop_percentage[2] / 100)

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_img.save(output_image_path)

percentage = 12.5
crop_image('VOTPP folder\Results\Plots\cluster_x-y_view.png', 'VOTPP folder\Results\Plots\cluster_x-y_view_cropped.png', (percentage, percentage, percentage, percentage))
crop_image('VOTPP folder\Results\Plots\cluster_x-z_view.png', 'VOTPP folder\Results\Plots\cluster_x-z_view_cropped.png', (percentage, percentage, percentage, percentage))
