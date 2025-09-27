import os

def gather_code(root="."):
    files_content = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".py"):  # 只讀 Python
                path = os.path.join(dirpath, f)
                with open(path, "r", encoding="utf-8") as fp:
                    files_content[path] = fp.read()
    return files_content

code_files = gather_code(".")
# code_files 是 dict: {檔案路徑: 檔案內容}
# print(code_files)  # 顯示前200字
import json
with open("code_files.json", "w", encoding="utf-8") as f:
    json.dump(code_files, f, ensure_ascii=False, indent=2)
