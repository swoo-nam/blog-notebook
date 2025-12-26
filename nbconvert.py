import shutil
import re
import pathlib

md_file_name = "2025-12-23-image_generation"
md_file_directory = "2025-12-23-image_generation_files"
created_at = "2025-12-23"

md = pathlib.Path(f"{md_file_name}.md")

src = pathlib.Path(
    f"{md_file_directory}/assets/images/posts/{created_at}"
)

dst = pathlib.Path(
    f"assets/images/posts/{created_at}"
)

# --- 이미지 이동 ---
if src.exists():
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.glob("*"):
        shutil.move(str(f), dst)

    shutil.rmtree(md_file_directory)

# --- Markdown 경로 보정 ---
text = md.read_text(encoding="utf-8")

# 1) nbconvert가 붙인 _files prefix 제거
text = re.sub(rf"{md_file_directory}/", "", text)

# 2) assets → /assets (웹 절대경로)
text = re.sub(r'!\[\]\((assets/)', r'![](/\1', text)

md.write_text(text, encoding="utf-8")

print("✅ Markdown image paths normalized to /assets/…")
