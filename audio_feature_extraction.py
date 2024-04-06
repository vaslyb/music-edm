import subprocess
import os

# Path to the directory containing input files
input_dir = 'audio/'

# Path to the directory where you want to save output files
output_dir = 'features/'

# Path to the executable file
exe_file = './streaming_extractor_music'

# Walk through the input directory recursively
for root, dirs, files in os.walk(input_dir):
    # Iterate over each file in the current directory
    for file in files:
        # Construct the full path of the input file
        input_path = os.path.join(root, file)
        
        # Construct the relative path of the input file
        relative_path = os.path.relpath(input_path, input_dir)
        
        # Construct the output directory structure based on the input directory structure
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)
        
        # Construct the output file path with the same name as the input file
        output_path = os.path.join(output_subdir, file)
        
        # Run the executable file with the input and output parameters
        subprocess.run([exe_file, input_path, output_path])

print("All files processed.")
