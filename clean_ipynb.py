import os
import nbformat

root = "."  # 要清理的資料夾
for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".ipynb"):
            path = os.path.join(dirpath, f)
            print("Cleaning:", path)
            with open(path, encoding="utf-8") as fp:
                nb = nbformat.read(fp, as_version=4)
            for cell in nb.cells:
                if "outputs" in cell:
                    cell["outputs"] = []
                if "execution_count" in cell:
                    cell["execution_count"] = None
            with open(path, "w", encoding="utf-8") as fp:
                nbformat.write(nb, fp)
