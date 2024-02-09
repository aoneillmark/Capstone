import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms

def create_spin_line(length=80, arrow_size=0.2, spacing=0.6, angle=0.5):
    """
    Creates an image of a vertical dotted line with arrows alternating between
    angles of 15 degrees and -15 degrees, saved as a PNG file with a transparent background.

    Parameters:
    length (int): Length of the line in terms of number of arrows.
    arrow_size (float): Length of each arrow.
    spacing (float): Spacing between arrows.
    angle (float): Angle of the arrows with respect to the vertical line.
    """

    # Create a figure with a transparent background
    fig, ax = plt.subplots(figsize=(2, length * spacing / 2))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # Generate arrow positions for a vertical line with alternating angles
    for i in range(length):
        x = 0  # All arrows on a straight vertical line
        y = i * spacing
        rotation_angle = angle if i % 2 == 0 else -angle

        # Create transformation for the arrow
        trans = mtransforms.Affine2D().rotate_deg_around(x, y, rotation_angle) + ax.transData
        
        # Add an arrow to the plot with rotation
        ax.arrow(x, y, 0, arrow_size, head_width=0.01, head_length=0.1, fc='k', ec='k', transform=trans)

    # Set limits and remove axes for clarity
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, length * spacing)
    ax.axis('off')

    # Save the figure with a transparent background
    plt.savefig('VOTPP folder\Results\Plots_editing\spin_line.tiff', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show()
    plt.close()



# Create the image
create_spin_line()
