from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_path, output_image_path, text, position, font_path, font_size):
    """
    Add text to an image.

    :param image_path: Path to the input image.
    :param output_image_path: Path to save the image with text.
    :param text: The text to add to the image.
    :param position: A tuple (x, y) to specify the position of the text.
    :param font_path: Path to the font file.
    :param font_size: Size of the font.
    """
    # Open the image
    with Image.open(image_path) as img:
        # Create drawing object
        draw = ImageDraw.Draw(img)

        # Define the font
        font = ImageFont.truetype(font_path, font_size)

        # Add text to image with black color
        draw.text(position, text, font=font, fill=(0, 0, 0))

        # Save the image
        img.save(output_image_path)

# Usage for Windows
times_new_roman_path = 'C:\\Windows\\Fonts\\times.ttf'  # Path to Times New Roman font on Windows
folder_path = 'VOTPP folder\\Results\\Plots_editing\\'
# add_text_to_image('VOTPP folder\\Results\\Plots\\cluster_x-y_view_cropped.png', 'VOTPP folder\\Results\\Plots\\cluster_x-y_view_with_text.png', 'A)', (10, 10), times_new_roman_path, 100)
# add_text_to_image('VOTPP folder\\Results\\Plots\\cluster_x-z_view_cropped.png', 'VOTPP folder\\Results\\Plots\\cluster_x-z_view_with_text.png', 'B)', (10, 10), times_new_roman_path, 100)

# add_text_to_image(folder_path + 'hahn1.png', folder_path + 'hahn1_text.png', 'A)', (10, 10), times_new_roman_path, 50)
# add_text_to_image(folder_path + 'hahn2.png', folder_path + 'hahn2_text.png', 'B)', (10, 10), times_new_roman_path, 50)
# add_text_to_image(folder_path + 'hahn3.png', folder_path + 'hahn3_text.png', 'C)', (10, 10), times_new_roman_path, 50)

# add_text_to_image(folder_path + 'sim1.png', folder_path + 'sim1_text.png', 'A)', (10, 10), times_new_roman_path, 25)
# add_text_to_image(folder_path + 'sim2.png', folder_path + 'sim2_text.png', 'B)', (10, 10), times_new_roman_path, 25)
# add_text_to_image(folder_path + 'sim3.png', folder_path + 'sim3_text.png', 'C)', (10, 10), times_new_roman_path, 25)

size = 100
pos_y = 75
add_text_to_image(folder_path + 'H_bath.png', folder_path + 'H_bath_text.png', 'A)', (10, pos_y), times_new_roman_path, size)
add_text_to_image(folder_path + 'H_dip.png', folder_path + 'H_dip_text.png', 'B)', (10, pos_y), times_new_roman_path, size)

add_text_to_image(folder_path + 'C_bath.png', folder_path + 'C_bath_text.png', 'A)', (10, pos_y), times_new_roman_path, size)
add_text_to_image(folder_path + 'C_dip.png', folder_path + 'C_dip_text.png', 'B)', (10, pos_y), times_new_roman_path, size)

add_text_to_image(folder_path + 'N_bath.png', folder_path + 'N_bath_text.png', 'A)', (10, pos_y), times_new_roman_path, size)
add_text_to_image(folder_path + 'N_dip.png', folder_path + 'N_dip_text.png', 'B)', (10, pos_y), times_new_roman_path, size)

add_text_to_image(folder_path + 'E_bath.png', folder_path + 'E_bath_text.png', 'A)', (10, pos_y), times_new_roman_path, size)
add_text_to_image(folder_path + 'E_dip.png', folder_path + 'E_dip_text.png', 'B)', (10, pos_y), times_new_roman_path, size)
