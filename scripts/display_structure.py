"""
Script to display project file structure and save to a text file
"""
import os

def display_structure(start_path, file_handle, indent=''):
    """
    Display the file structure of a directory and write to a file
    
    Parameters:
        start_path: Path to the directory to display
        file_handle: File handle to write the output to
        indent: Indentation string for formatting (used in recursion)
    """
    # Print the root directory
    if indent == '':
        file_handle.write(f"\nProject structure for: {os.path.basename(start_path)}\n")
        file_handle.write("=" * 50 + "\n")
    
    # Get all items in the directory
    try:
        items = os.listdir(start_path)
    except PermissionError:
        file_handle.write(f"{indent}[Permission denied]\n")
        return
    
    # Filter out unwanted files/directories
    exclude = ['.git', '__pycache__', '.vscode', '.idea', 'venv', 'env', '.ipynb_checkpoints']
    items = [item for item in items if item not in exclude]
    
    # Sort: directories first, then files
    dirs = [item for item in items if os.path.isdir(os.path.join(start_path, item))]
    files = [item for item in items if os.path.isfile(os.path.join(start_path, item))]
    
    dirs.sort()
    files.sort()
    
    # Print directories
    for d in dirs:
        full_path = os.path.join(start_path, d)
        file_handle.write(f"{indent}📁 {d}/\n")
        display_structure(full_path, file_handle, indent + '  ')
    
    # Print files
    for f in files:
        # Get file size
        full_path = os.path.join(start_path, f)
        size_kb = os.path.getsize(full_path) / 1024
        
        # Use different icons based on file type
        if f.endswith(('.py')):
            icon = "🐍"
        elif f.endswith(('.txt', '.md', '.csv')):
            icon = "📄"
        elif f.endswith(('.jpg', '.png', '.gif')):
            icon = "🖼️"
        elif f.endswith(('.json', '.yaml', '.yml')):
            icon = "📋"
        else:
            icon = "📄"
        
        file_handle.write(f"{indent}{icon} {f} ({size_kb:.1f} KB)\n")

if __name__ == "__main__":
    # Use the current directory as the starting point
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to output file
    output_file = os.path.join(output_dir, "project_structure.txt")
    
    # Write structure to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Project Structure Generated on: {os.path.basename(project_root)}\n")
        display_structure(project_root, f)
    
    print(f"Project structure has been saved to: {output_file}")