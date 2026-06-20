import os
import fitz

knowledge_dir = "knowledge"
files = os.listdir(knowledge_dir)
pdf_files = [f for f in files if f.endswith(".pdf")]
pdf_path = os.path.join(knowledge_dir, pdf_files[0])

doc = fitz.open(pdf_path)
page_width = doc[0].rect.width
page_height = doc[0].rect.height
mid = page_width / 2

print(f"页面尺寸: {page_width:.0f} x {page_height:.0f} pt")
print(f"A3 横向: 1191 x 842, A4: 595 x 842")
print(f"总页数: {len(doc)}\n")

for p in range(min(30, len(doc))):
    page = doc[p]
    text = page.get_text("text").strip()
    text_len = len(text)

    blocks = page.get_text("dict")["blocks"]
    text_blocks = [b for b in blocks if b["type"] == 0]
    image_blocks = [b for b in blocks if b["type"] == 1]

    # 双栏粗略检测
    left_count = 0
    right_count = 0
    for b in text_blocks:
        x = (b["bbox"][0] + b["bbox"][2]) / 2
        t_len = sum(len(s["text"].strip()) for line in b["lines"] for s in line["spans"])
        if t_len > 20:
            if x < mid:
                left_count += t_len
            else:
                right_count += t_len

    # 统计字号
    sizes = set()
    for b in text_blocks:
        for line in b["lines"]:
            for s in line["spans"]:
                sizes.add(round(s["size"], 1))

    dual_hint = "← 可能双栏" if left_count > 200 and right_count > 200 else ""

    print(f"页 {p+1:3d}: {text_len:5d}字, 文本块{len(text_blocks):2d}, "
          f"图片{len(image_blocks):2d}, "
          f"左{left_count:5d}右{right_count:5d} {dual_hint}")

doc.close()
