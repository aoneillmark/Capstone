import nbformat as nbf

# Read the input text
with open("input.txt", "r") as f:
    data = f.read()

# Split the data into cells based on the markers
cells = data.split("---MARKDOWN---")
cells = [cell.split("---CODE---") for cell in cells if cell.strip()]

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
for cell in cells:
    if len(cell) == 2:
        nb.cells.append(nbf.v4.new_markdown_cell(cell[0].strip()))
        nb.cells.append(nbf.v4.new_code_cell(cell[1].strip()))
    elif cell[0].strip():
        nb.cells.append(nbf.v4.new_markdown_cell(cell[0].strip()))

# Write the notebook to a file
with open("output.ipynb", "w") as f:
    nbf.write(nb, f)
