import os
from PIL import Image as PILImage
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

def save_and_show_graph(graph, filename: str, show_image: bool = False, 
                       background_color: str = "white", padding: int = 10):
    """
    Saves the graph image using langchain's built-in draw_mermaid_png function.

    Args:
        graph: The compiled graph object.
        filename: The custom name for the image file (without extension).
        show_image: Boolean flag to display the image after saving.
        background_color: Background color of the image. Defaults to "white".
        padding: Padding around the image. Defaults to 10.
    """
    # Ensure the GraphImages directory exists
    os.makedirs("GraphImages", exist_ok=True)

    # Get the mermaid syntax from the graph
    mermaid_syntax = graph.get_graph().draw_mermaid()
    
    # Define the output path
    img_path = f"./GraphImages/{filename}.png"
    
    # Use the built-in draw_mermaid_png function
    png_bytes = draw_mermaid_png(
        mermaid_syntax=mermaid_syntax,
        output_file_path=img_path,
        background_color=background_color,
        padding=padding
    )
    
    print(f"Graph saved as '{img_path}'")

    # Open and optionally display the image
    if show_image:
        img = PILImage.open(img_path)
        img.show()
        
    return png_bytes