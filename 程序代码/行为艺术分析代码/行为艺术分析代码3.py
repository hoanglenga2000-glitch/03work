import pdfplumber
import os


def read_full_pdf_to_txt(pdf_path, output_txt_path):
    """
    完整读取PDF所有页面文字，保存到桌面TXT文件（保留页码和原始格式）
    :param pdf_path: PDF文件完整路径（你的桌面PDF路径）
    :param output_txt_path: 桌面TXT输出路径
    """
    # 1. 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：PDF文件不存在！路径：\n{pdf_path}")
        print("⚠️  提示：确认PDF文件名是否为“行为艺术现场.pdf”（含后缀.pdf），且在桌面路径下")
        return

    # 2. 读取PDF所有页面文字（保留原始格式，标注页码）
    print(f"🔍 正在读取PDF全部内容：{pdf_path}")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        # 遍历所有页面（从第1页到最后一页，页码从1开始标注）
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            # 提取当前页面文字（strip=False保留空格、换行等原始格式）
            page_text = page.extract_text(strip=False)
            if page_text:
                # 标注页码（方便后续对照PDF原文）
                full_text += f"==================================================\n"
                full_text += f"📄 第{page_num}页 / 共{total_pages}页\n"
                full_text += f"==================================================\n"
                full_text += page_text + "\n\n"  # 页面间加空行，避免内容粘连
            # 打印进度（每10页提示一次，方便了解读取进度）
            if page_num % 10 == 0 or page_num == total_pages:
                print(f"✅ 已读取第{page_num}页，剩余{total_pages - page_num}页...")

    # 3. 检查是否成功读取到文字
    if not full_text:
        print(f"❌ 错误：未从PDF中提取到文字内容！")
        print(
            "⚠️  排查：1. 确认PDF非扫描件（扫描件需先OCR识别）；2. 尝试更新pdfplumber版本（pip install --upgrade pdfplumber）")
        return

    # 4. 写入TXT文件（UTF-8编码，避免中文乱码）
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        # 头部说明（标注来源、提取时间、格式说明）
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"《行为艺术现场》（蔡青著）- 完整文字提取\n" \
                 f"=============================================\n" \
                 f"提取来源：{pdf_path}\n" \
                 f"提取时间：{current_time}\n" \
                 f"格式说明：1. 每页标注页码（共{total_pages}页）；2. 保留PDF原始段落换行和空格；3. 页面间用分隔线区分\n" \
                 f"=============================================\n\n"
        txt_file.write(header + full_text)

    print(f"\n🎉 成功读取全部PDF内容！TXT文件路径：\n{output_txt_path}")
    print(f"📊 提取统计：共{total_pages}页，文字总量约{len(full_text)}字符")


# -------------------------- 你的文件路径（无需修改，已适配桌面PDF）--------------------------
if __name__ == "__main__":
    # 1. 输入：你的PDF路径（D盘桌面，文件名：行为艺术现场.pdf）
    PDF_FULL_PATH = r"D:\桌面\行为艺术现场.pdf"
    # 2. 输出：桌面TXT文件（固定名：行为艺术现场_完整全文.txt）
    OUTPUT_TXT_PATH = r"D:\桌面\行为艺术现场_完整全文.txt"

    # 执行完整读取
    read_full_pdf_to_txt(pdf_path=PDF_FULL_PATH, output_txt_path=OUTPUT_TXT_PATH)