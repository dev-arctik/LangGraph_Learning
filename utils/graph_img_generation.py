import os
from PIL import Image as PILImage

def save_and_show_graph(graph, filename: str, show_image: bool = False):
    """
    Saves the graph image and optionally displays it.

    Args:
        graph: The compiled graph object.
        filename: The custom name for the image file (without extension).
        show_image: Boolean flag to display the image after saving.
    """
    # Ensure the GraphImages directory exists
    os.makedirs("GraphImages", exist_ok=True)

    # Save the graph image as a PNG file in the GraphImages directory
    img_path = f"./GraphImages/{filename}.png"
    with open(img_path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    print(f"Graph saved as '{img_path}'")

    # Open and optionally display the image
    if show_image:
        img = PILImage.open(img_path)
        img.show()
