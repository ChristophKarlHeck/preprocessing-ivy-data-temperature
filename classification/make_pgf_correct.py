import re
import glob

# Get all .pgf files
pgf_files = glob.glob("*.pgf")  

for file in pgf_files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix \mathdefault{}
    content_fixed = re.sub(r"\\mathdefault{([^}]*)}", r"\1", content)

    with open(file, "w", encoding="utf-8") as f:
        f.write(content_fixed)

    print(f"Fixed: {file}")