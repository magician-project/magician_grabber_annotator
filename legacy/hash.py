import os

def get_md5(file_path):
    # Construct the command
    command = f"md5sum {file_path}"
    
    # Execute the command and capture the output
    output = os.popen(command).read()
    
    # Parse the output to extract the MD5 hash
    md5_hash = output.split()[0]
    
    return md5_hash

# Example usage
file_path = "tmp.png"
md5_hash = get_md5(file_path)
print("MD5 Hash:", md5_hash)

