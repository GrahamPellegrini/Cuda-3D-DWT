import os

def remove_zone_identifiers(directory):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file contains ':Zone.Identifier'
            if ':Zone.Identifier' in file:
                try:
                    # Remove the file
                    os.remove(os.path.join(root, file))
                    # Print a message indicating the file has been removed
                    print(f"Removed: {os.path.join(root, file)}")
                except Exception as e:
                    print(f"Failed to remove {os.path.join(root, file)}: {e}")

# Example usage
remove_zone_identifiers("/home/grahampelle/cce3015/code/assignment-1/CHAOS-Test-Sets")