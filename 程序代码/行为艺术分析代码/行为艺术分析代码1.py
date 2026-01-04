import pandas as pd
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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


class WikipediaDataAnalyzer:
    def __init__(self, file_path):
        """
        初始化分析器
        """
        self.file_path = file_path
        self.raw_text = ""
        self.paragraphs = []
        self.words = []
        self.load_data()

    def load_data(self):
        """
        加载数据文件
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()

            # 提取段落
            lines = self.raw_text.split('\n')
            self.paragraphs = [line.strip() for line in lines if len(line.strip()) > 10]

            print(f"成功加载数据: {len(self.paragraphs)} 个段落")

        except Exception as e:
            print(f"加载数据失败: {e}")

    def clean_text(self, text):
        """
        清洗文本，移除标点符号和数字
        """
        # 移除标点符号和数字
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def segment_text(self):
        """
        对文本进行分词
        """
        if not self.paragraphs:
            print("没有可分析的段落")
            return

        # 合并所有段落
        full_text = ' '.join(self.paragraphs)
        # 清洗文本
        cleaned_text = self.clean_text(full_text)
        # 使用jieba进行分词
        self.words = [word for word in jieba.cut(cleaned_text) if len(word) > 1]

        print(f"分词完成，共 {len(self.words)} 个词语")

    def basic_statistics(self):
        """
        基础统计分析
        """
        if not self.paragraphs:
            print("没有可分析的段落")
            return

        stats = {
            '段落数量': len(self.paragraphs),
            '总字符数': sum(len(p) for p in self.paragraphs),
            '平均段落长度': sum(len(p) for p in self.paragraphs) / len(self.paragraphs),
            '最长段落': max(len(p) for p in self.paragraphs),
            '最短段落': min(len(p) for p in self.paragraphs)
        }

        # 如果有分词结果，添加词汇统计
        if self.words:
            stats['词汇总数'] = len(self.words)
            stats['独特词汇'] = len(set(self.words))
            stats['词汇密度'] = stats['独特词汇'] / stats['词汇总数'] if stats['词汇总数'] > 0 else 0

        return stats

    def word_frequency_analysis(self, top_n=20):
        """
        词频分析
        """
        if not self.words:
            print("请先进行分词")
            return

        # 计算词频
        word_freq = Counter(self.words)
        top_words = word_freq.most_common(top_n)

        return top_words

    def extract_keywords(self, top_n=10):
        """
        提取关键词
        """
        if not self.paragraphs:
            print("没有可分析的段落")
            return

        # 合并所有段落
        full_text = ' '.join(self.paragraphs)

        # 使用TF-IDF算法提取关键词
        keywords_tfidf = jieba.analyse.extract_tags(
            full_text,
            topK=top_n,
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'a')  # 名词、动词、形容词
        )

        # 使用TextRank算法提取关键词
        keywords_textrank = jieba.analyse.textrank(
            full_text,
            topK=top_n,
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'a')
        )

        return {
            'TF-IDF': keywords_tfidf,
            'TextRank': keywords_textrank
        }

    def create_visualizations(self, top_words):
        """
        创建可视化图表
        """
        if not top_words:
            print("没有词频数据")
            return

        # 创建输出目录
        output_dir = "analysis_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 词频柱状图
        plt.figure(figsize=(12, 8))
        words, counts = zip(*top_words)
        plt.bar(words, counts, color='skyblue')
        plt.title('词频统计（前20名）')
        plt.xlabel('词语')
        plt.ylabel('出现次数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，避免在非交互环境中显示

        # 2. 词云图
        if self.words:
            word_freq_dict = dict(top_words)
            font_path = find_chinese_font()
            wordcloud_params = {
                'width': 800,
                'height': 600,
                'background_color': 'white',
                'max_words': 100
            }
            if font_path:
                wordcloud_params['font_path'] = font_path
            
            wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(word_freq_dict)

            plt.figure(figsize=(10, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('词云图')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免在非交互环境中显示

        # 3. 段落长度分布
        if self.paragraphs:
            plt.figure(figsize=(10, 6))
            para_lengths = [len(p) for p in self.paragraphs]
            plt.hist(para_lengths, bins=20, color='lightgreen', alpha=0.7)
            plt.title('段落长度分布')
            plt.xlabel('段落长度（字符数）')
            plt.ylabel('段落数量')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/paragraph_lengths.png', dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免在非交互环境中显示

    def save_analysis_results(self, stats, top_words, keywords):
        """
        保存分析结果到文件
        """
        output_dir = "analysis_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存统计分析
        with open(f'{output_dir}/statistics.txt', 'w', encoding='utf-8') as f:
            f.write("=== 文本统计分析 ===\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

            f.write("\n=== 高频词汇 ===\n\n")
            for word, count in top_words:
                f.write(f"{word}: {count}次\n")

            f.write("\n=== 关键词提取 ===\n\n")
            f.write("TF-IDF算法:\n")
            for word, weight in keywords['TF-IDF']:
                f.write(f"{word}: {weight:.4f}\n")

            f.write("\nTextRank算法:\n")
            for word, weight in keywords['TextRank']:
                f.write(f"{word}: {weight:.4f}\n")

        # 保存为CSV文件
        df_stats = pd.DataFrame([stats])
        df_stats.to_csv(f'{output_dir}/statistics.csv', index=False, encoding='utf-8-sig')

        df_words = pd.DataFrame(top_words, columns=['词语', '频次'])
        df_words.to_csv(f'{output_dir}/word_frequency.csv', index=False, encoding='utf-8-sig')

        print(f"分析结果已保存到 {output_dir} 目录")

    def comprehensive_analysis(self):
        """
        执行全面分析
        """
        print("开始数据分析...")

        # 分词
        self.segment_text()

        # 基础统计
        print("\n1. 基础统计分析:")
        stats = self.basic_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 词频分析
        print("\n2. 词频分析 (前20名):")
        top_words = self.word_frequency_analysis(20)
        for i, (word, count) in enumerate(top_words, 1):
            print(f"  {i:2d}. {word}: {count}次")

        # 关键词提取
        print("\n3. 关键词提取:")
        keywords = self.extract_keywords(10)

        print("  TF-IDF算法:")
        for i, (word, weight) in enumerate(keywords['TF-IDF'], 1):
            print(f"    {i:2d}. {word}: {weight:.4f}")

        print("  TextRank算法:")
        for i, (word, weight) in enumerate(keywords['TextRank'], 1):
            print(f"    {i:2d}. {word}: {weight:.4f}")

        # 创建可视化
        print("\n4. 生成可视化图表...")
        self.create_visualizations(top_words)

        # 保存结果
        print("\n5. 保存分析结果...")
        self.save_analysis_results(stats, top_words, keywords)

        print("\n分析完成!")


# 主程序
if __name__ == "__main__":
    # 确保安装了必要的库
    try:
        import jieba
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行以下命令安装:")
        print("pip install jieba matplotlib wordcloud seaborn pandas")
        exit(1)

    # 数据文件路径 - 请根据你的实际文件路径修改
    data_file = "行为艺术_爬取结果.txt"

    # 如果文件不存在，尝试其他可能的文件名
    if not os.path.exists(data_file):
        possible_files = [
            "行为艺术_wikipedia.txt",
            "行为艺术_内容.txt",
            "行为艺术_爬取结果.txt"
        ]
        for file in possible_files:
            if os.path.exists(file):
                data_file = file
                break
        else:
            print("找不到数据文件，请确保文件存在并修改data_file变量")
            exit(1)

    # 创建分析器并执行分析
    analyzer = WikipediaDataAnalyzer(data_file)
    analyzer.comprehensive_analysis()