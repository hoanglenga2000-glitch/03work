# 这是一个概念性的代码逻辑，展示如何生成更高级的词云
# 假设您已经安装了 stylecloud: pip install stylecloud

import os
import sys

# 检查并安装必要的库
try:
    import stylecloud
    import jieba
except ImportError as e:
    print(f"缺少必要的库: {e}")
    print("请运行以下命令安装:")
    print("pip install stylecloud jieba")
    sys.exit(1)


def find_chinese_font():
    """查找可用的中文字体路径"""
    import platform
    
    # Windows系统字体路径
    if platform.system() == 'Windows':
        font_paths = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\msyh.ttc',    # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
        ]
    else:
        # Linux/Mac系统字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/System/Library/Fonts/PingFang.ttc',  # Mac
        ]
    
    # 检查字体文件是否存在
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    # 如果找不到，返回None
    return None

# 1. 准备文本数据：这里我们要区分两类文本
# 文本A：书中关于"身体/痛苦/对抗"的描述（如东村部分）
text_pain = """
自虐 烙铁 鲜血 苍蝇 窒息 捆绑 伤口 忍受 冰冷 尸体 
剧痛 极限 压抑 愤怒 地下 审视 荒诞 坚硬 铁链 玻璃
""" # 示例关键词，实际应从书中提取

# 文本B：书中关于"治愈/自然/互动"的描述（如东南亚/治愈理论部分）
text_healing = """
治愈 心灵 温暖 拥抱 沟通 泥土 呼吸 生长 流动 
祈祷 冥想 连接 观众 游戏 欢乐 海洋 树叶 融合 
日常 吃饭 节日 朋友 协作 和平
""" # 示例关键词

# 可选：从文本文件中读取并提取关键词
def extract_keywords_from_file(file_path, keywords_list):
    """从文件中提取包含特定关键词的文本"""
    try:
        if os.path.exists(file_path):
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            text = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text:
                # 使用jieba分词并提取关键词
                words = jieba.cut(text)
                # 过滤出包含关键词的文本片段
                relevant_words = [w for w in words if any(kw in w for kw in keywords_list)]
                return ' '.join(relevant_words)
    except Exception as e:
        print(f"读取文件时出错: {e}")
    return None

# 如果存在文本文件，可以尝试从文件中提取
# text_file = r"D:\桌面\行为艺术现场_完整全文.txt"
# if os.path.exists(text_file):
#     pain_keywords = ['痛苦', '自虐', '暴力', '死亡', '压抑', '愤怒', '对抗']
#     healing_keywords = ['治愈', '温暖', '自然', '互动', '沟通', '和平', '欢乐']
#     extracted_pain = extract_keywords_from_file(text_file, pain_keywords)
#     extracted_healing = extract_keywords_from_file(text_file, healing_keywords)
#     if extracted_pain:
#         text_pain = extracted_pain
#     if extracted_healing:
#         text_healing = extracted_healing

# 2. 生成"身体/对抗"主题的词云 - 使用人体图标，深红色调
try:
    font_path = find_chinese_font()
    stylecloud_params_pain = {
        'text': text_pain,
        'icon_name': 'fas fa-user-injured',  # 使用受伤的人体图标
        'palette': 'cmocean.sequential.Amp_10',  # 深色/强烈的配色
        'background_color': 'black',  # 黑色背景增强压抑感
        'output_name': 'pain_cloud.png'
    }
    if font_path:
        stylecloud_params_pain['font_path'] = font_path
    
    stylecloud.gen_stylecloud(**stylecloud_params_pain)
    print("✓ 成功生成痛苦主题词云: pain_cloud.png")
except Exception as e:
    print(f"✗ 生成痛苦主题词云失败: {e}")
    print("提示: 如果缺少图标库，可以尝试使用普通词云库")

# 3. 生成"治愈/自然"主题的词云 - 使用手掌或树叶图标，清新色调
try:
    font_path = find_chinese_font()
    stylecloud_params_healing = {
        'text': text_healing,
        'icon_name': 'fas fa-hands-helping',  # 使用互助的手图标
        'palette': 'cmocean.sequential.Algae_10',  # 绿色/自然配色
        'background_color': 'white',
        'output_name': 'healing_cloud.png'
    }
    if font_path:
        stylecloud_params_healing['font_path'] = font_path
    
    stylecloud.gen_stylecloud(**stylecloud_params_healing)
    print("✓ 成功生成治愈主题词云: healing_cloud.png")
except Exception as e:
    print(f"✗ 生成治愈主题词云失败: {e}")
    print("提示: 如果缺少图标库，可以尝试使用普通词云库")

print("高级词云生成完毕。这两张图对比可以直观展示《行为艺术现场》中从对抗到治愈的跨度。")
