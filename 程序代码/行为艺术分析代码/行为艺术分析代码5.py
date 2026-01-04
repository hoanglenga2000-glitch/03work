# ==================== 1. 导入必要的库 ====================
import os
import re
import jieba
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from datetime import datetime
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# 中文自然语言处理相关
from snownlp import SnowNLP
import jieba.posseg as pseg


# ==================== 字体工具函数 ====================
def find_chinese_font():
    """查找可用的中文字体路径"""
    import platform
    import os
    
    # Windows系统字体路径
    if platform.system() == 'Windows':
        font_paths = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\msyh.ttc',    # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
            r'C:\Windows\Fonts\msyhbd.ttc',  # 微软雅黑 Bold
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
    
    # 如果找不到，返回None（WordCloud会使用默认字体）
    return None

# 网络分析相关（可选，用于未来扩展）
# import networkx as nx
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats


# ==================== 2. 数据预处理类（优化版） ====================
class TextPreprocessor:
    """文本预处理类（优化版）"""

    def __init__(self):
        self.load_custom_dict()
        self.load_stopwords()

    def load_custom_dict(self):
        """加载行为艺术专业术语词典（增强版）"""
        art_terms = [
            # 艺术类型
            '行为艺术', '现场艺术', '表演艺术', '身体艺术', '实验艺术', '当代艺术', '观念艺术',
            '装置艺术', '互动艺术', '参与式艺术', '社会介入艺术', '社区艺术',

            # 艺术节/活动
            '谷雨行动', 'UP-ON', 'OPEN国际', '国际黑市', '出入天堂', '泛亚洲', '亚洲顶峰',
            'IN:ACT', '现场纪实', '艺术现场', '艺术节', '双年展', '三年展', '艺术博览会',

            # 艺术家
            '周斌', '蔡青', '李文', '幸鑫', '陈进', '相西石', '刘成英', '丹羽良德',
            '白井广美', '王楚禹', '何玲', '林荣华', '马拉帝', '舒阳', '金光哲',
            '春朋·阿普宿克', '霜田诚二', '谢德庆', '万巧', '卓静', '孙寒宵',

            # 地点/机构
            '浓园', '北村艺术区', '新都', '汶川纪念碑', '成都蓝顶', '798艺术区',
            '美术馆', '艺术中心', '画廊', '工作室', '艺术社区', '创意园区',

            # 专业术语
            '策展人', '策展', '展览', '参展', '观展', '观众', '互动', '参与',
            '表演', '作品', '创作', '实践', '实验', '探索', '突破', '创新',
            '身体性', '现场性', '时间性', '空间性', '物质性', '观念性',
            '批判性', '反思性', '社会性', '政治性', '文化性'
        ]

        for term in art_terms:
            jieba.add_word(term, freq=1000, tag='nz')

    def load_stopwords(self):
        """加载停用词表"""
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '但', '从', '想', '做',
            '来', '把', '又', '面', '么', '两', '一些', '已经', '没有', '这个',
            '这种', '这样', '这么', '那么', '因为', '所以', '如果', '虽然',
            '但是', '然后', '而且', '或者', '因为', '所以', '然后', '之后',
            '之前', '时候', '一些', '一点', '一下', '一起', '一种', '一样',
            '一般', '一直', '一下', '一点', '一些', '一样', '一般', '一直'
        ])

    def clean_text(self, text):
        """清理文本（增强版）"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\w\u4e00-\u9fff\s,.!?;:，。！？；：\-—、（）《》【】「」『』]', '', text)
        # 统一空格
        text = re.sub(r'\s+', ' ', text)
        # 移除数字编号（如1. 2. 3.）
        text = re.sub(r'\b\d+[\.、]\s*', '', text)
        return text.strip()

    def split_by_festival_v2(self, text):
        """改进的艺术节分割方法"""
        festivals = []
        lines = text.split('\n')

        # 定义更精确的艺术节开始模式
        festival_patterns = [
            r'^.*行为艺术节.*$',
            r'^.*艺术节.*现场纪实.*$',
            r'^.*国际黑市.*$',
            r'^.*谷雨行动.*$',
            r'^.*UP-ON.*$',
            r'^.*OPEN.*$',
            r'^.*\d{4}年.*行为艺术.*$',
            r'^.*[《》].*[》].*$',  # 包含书名号的标题
            r'^.*表演.*作品.*$',
            r'^.*\d{4}年\d{1,2}月.*日.*$'  # 日期开头的行
        ]

        current_festival = []
        current_title = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_title = False
            for pattern in festival_patterns:
                if re.match(pattern, line) and len(line) < 150:
                    is_title = True
                    break

            # 包含特定关键词的短行也可能是标题
            title_keywords = ['艺术节', '现场', '纪实', '表演', '作品', '展览', '展示']
            if any(keyword in line for keyword in title_keywords) and len(line) < 100:
                is_title = True

            if is_title:
                if current_festival and len('\n'.join(current_festival)) > 500:
                    festivals.append({
                        'title': current_title,
                        'content': '\n'.join(current_festival)
                    })
                current_festival = [line]
                current_title = line
            else:
                current_festival.append(line)

        # 处理最后一个艺术节
        if current_festival and len('\n'.join(current_festival)) > 500:
            festivals.append({
                'title': current_title,
                'content': '\n'.join(current_festival)
            })

        return festivals if festivals else [{'title': '行为艺术文本', 'content': text}]

    def extract_metadata_v2(self, text):
        """增强的元数据提取"""
        metadata = {
            'years': [],
            'locations': [],
            'artists': [],
            'works': [],
            'keywords': [],
            'dates': []
        }

        # 提取年份（更全面的匹配）
        year_patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})[日号]',
            r'(\d{4})年(\d{1,2})月',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})'
        ]

        for pattern in year_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    year = match[0]
                    if year not in metadata['years']:
                        metadata['years'].append(year)

        # 提取地点（扩展列表）
        locations_db = {
            '中国': ['北京', '上海', '广州', '深圳', '成都', '重庆', '杭州', '南京',
                     '武汉', '西安', '长沙', '沈阳', '青岛', '厦门', '苏州', '天津',
                     '香港', '澳门', '台湾', '台北', '高雄', '台中'],
            '亚洲': ['东京', '大阪', '京都', '首尔', '釜山', '新加坡', '曼谷', '清迈',
                     '河内', '胡志明市', '马尼拉', '雅加达', '吉隆坡', '新德里', '孟买'],
            '欧美': ['纽约', '洛杉矶', '伦敦', '巴黎', '柏林', '罗马', '米兰', '巴塞罗那',
                     '莫斯科', '悉尼', '墨尔本', '多伦多', '温哥华']
        }

        found_locations = []
        for country, cities in locations_db.items():
            for city in cities:
                if city in text and city not in found_locations:
                    found_locations.append(city)
            if country in text and country not in found_locations:
                found_locations.append(country)

        metadata['locations'] = found_locations

        # 提取艺术家（使用词性标注）
        words = pseg.cut(text)
        artists = []
        for word, flag in words:
            if flag == 'nr' and len(word) >= 2 and len(word) <= 4:
                artists.append(word)

        # 去重并限制数量
        metadata['artists'] = list(set(artists))[:15]

        # 提取作品名称
        work_patterns = [
            r'《([^》]{2,50})》',
            r'作品[《"]([^》"]{2,50})[》"]',
            r'表演[《"]([^》"]{2,50})[》"]',
            r'"([^"]{2,50})"表演',
            r"'([^']{2,50})'作品"
        ]

        works = []
        for pattern in work_patterns:
            matches = re.findall(pattern, text)
            works.extend(matches)

        metadata['works'] = list(set(works))[:10]

        # 提取关键词（TF-IDF简单版）
        words = jieba.lcut(text)
        words = [w for w in words if len(w) > 1 and w not in self.stopwords]
        word_freq = Counter(words)
        metadata['keywords'] = [word for word, freq in word_freq.most_common(20)]

        return metadata


# ==================== 3. 情感分析引擎（优化版） ====================
class ArtTextSentimentAnalyzerV2:
    """艺术文本情感分析器（优化版）"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_lexicon = self.load_enhanced_sentiment_lexicon()
        self.setup_sentiment_rules()

    def load_enhanced_sentiment_lexicon(self):
        """加载增强的情感词典"""
        return {
            'positive': [
                '成功', '精彩', '感动', '热烈', '兴奋', '愉快', '欣赏', '赞美', '掌声', '欢笑',
                '创新', '突破', '深刻', '震撼', '温馨', '和谐', '自由', '解放', '激情', '活力',
                '欣慰', '信心', '难得', '幽默', '机智', '肃穆', '诗意', '赏心悦目', '可歌可泣',
                '美好', '深入', '圆满', '欢乐', '真诚', '快乐', '热情', '执着', '美丽', '善良',
                '朴实', '简单', '享受', '舒服', '有趣', '宏伟', '期待', '希望', '预祝', '越来越好',
                '精彩纷呈', '和谐共处', '印象深刻', '受益匪浅', '大开眼界', '耳目一新', '流连忘返',
                '赞不绝口', '好评如潮', '反响热烈', '座无虚席', '引人入胜', '发人深省', '意义深远'
            ],
            'negative': [
                '痛苦', '艰难', '沉重', '悲哀', '愤怒', '暴力', '死亡', '压抑', '冲突', '争议',
                '危险', '困难', '挑战', '孤独', '冷漠', '批判', '质疑', '反抗', '束缚', '限制',
                '凄惨', '沉重', '累坏', '无聊', '弱智', '烦腻', '不自知', '寡淡无味', '混乱',
                '发泄', '无奈', '可怕', '失忆', '惶恐', '凄凉', '艰苦', '挣扎', '闹剧', '绝望',
                '胡闹', '惊险', '紧张', '笨拙', '担心', '死亡', '悲伤', '屠杀', '自杀', '危机',
                '抗议', '炸弹', '堵截', '关闭', '失落', '笨', '冷淡', '尴尬', '恐惧', '恶化',
                '阴暗', '战争', '暴力', '伤痛', '困难', '挑战', '危险', '争议', '冲突', '失败',
                '失望', '遗憾', '缺陷', '不足', '问题', '错误', '批评', '指责', '反对', '抵制'
            ],
            'experimental': [
                '实验', '探索', '尝试', '冒险', '先锋', '前卫', '创新', '突破', '挑战',
                '探索性', '实验性', '先锋性', '前卫性', '创新性', '突破性', '尝试性',
                '大胆', '新颖', '独特', '前所未有', '史无前例', '开创性', '探索精神'
            ],
            'critical': [
                '批判', '质疑', '反思', '挑战', '反抗', '颠覆', '讽刺', '抗议', '争议',
                '批判性', '反思性', '质疑性', '挑战性', '反抗性', '颠覆性', '讽刺性',
                '揭露', '暴露', '质问', '追问', '深思', '反省', '检讨', '审视'
            ],
            'interactive': [
                '互动', '参与', '交流', '对话', '合作', '共享', '观众', '邀请', '配合',
                '互动性', '参与性', '交流性', '对话性', '合作性', '共享性', '共同',
                '一起', '协同', '协作', '联手', '携手', '联合', '集体', '群体'
            ],
            'political': [
                '政治', '抗议', '示威', '独裁', '自由', '民主', '权力', '压迫', '革命',
                '政治性', '社会性', '权利', '平等', '公平', '正义', '体制', '制度',
                '政府', '国家', '民族', '阶级', '身份', '权利', '解放', '抗争'
            ],
            'ritual': [
                '仪式', '祭奠', '纪念', '哀悼', '祈祷', '神圣', '庄严', '肃穆', '葬礼',
                '仪式感', '仪式性', '祭祀', '祭拜', '缅怀', '追思', '怀念', '纪念性'
            ],
            'aesthetic': [
                '美', '美学', '审美', '艺术性', '美感', '美丽', '优美', '优雅', '精致',
                '细腻', '粗犷', '简约', '繁复', '抽象', '具象', '形式', '色彩', '线条'
            ]
        }

    def setup_sentiment_rules(self):
        """设置情感分析规则"""
        # 情感强度调节因子
        self.intensity_factors = {
            '非常': 2.0, '很': 1.5, '特别': 1.8, '极其': 2.0, '十分': 1.5,
            '相当': 1.3, '比较': 1.2, '有点': 0.8, '稍微': 0.7, '略微': 0.7,
            '不太': 0.5, '不': 0.3, '没': 0.3, '毫无': 0.1, '完全没有': 0.1
        }

    def calculate_sentiment_score_v2(self, text):
        """改进的情感分数计算"""
        try:
            # 1. SnowNLP基础得分
            s = SnowNLP(text)
            snow_score = s.sentiments

            # 2. 基于词典的情感分析
            words = jieba.lcut(text)
            total_words = max(len(words), 1)

            # 计算正面和负面词
            pos_words = [w for w in words if w in self.sentiment_lexicon['positive']]
            neg_words = [w for w in words if w in self.sentiment_lexicon['negative']]

            # 考虑情感强度修饰词
            enhanced_pos = len(pos_words)
            enhanced_neg = len(neg_words)

            for i, word in enumerate(words):
                if i > 0 and words[i - 1] in self.intensity_factors:
                    factor = self.intensity_factors[words[i - 1]]
                    if word in pos_words:
                        enhanced_pos += factor - 1.0
                    elif word in neg_words:
                        enhanced_neg += factor - 1.0

            # 3. 计算基于词典的得分
            if enhanced_pos + enhanced_neg > 0:
                lexicon_score = enhanced_pos / (enhanced_pos + enhanced_neg)
            else:
                lexicon_score = 0.5

            # 4. 综合得分（加权平均）
            # SnowNLP权重0.6，词典权重0.4
            combined_score = snow_score * 0.6 + lexicon_score * 0.4

            # 5. 考虑否定词的影响
            negation_words = ['不', '没', '无', '非', '未', '否']
            negation_count = sum(1 for w in words if w in negation_words)
            negation_effect = 1.0 - (negation_count / total_words * 0.1)  # 最多影响10%
            combined_score *= negation_effect

            # 确保分数在0-1之间
            final_score = max(0.0, min(1.0, combined_score))

            return {
                'overall': round(final_score, 3),
                'snownlp': round(snow_score, 3),
                'lexicon': round(lexicon_score, 3),
                'pos_words': len(pos_words),
                'neg_words': len(neg_words),
                'negation_count': negation_count
            }

        except Exception as e:
            print(f"情感计算错误: {e}")
            return {
                'overall': 0.5,
                'snownlp': 0.5,
                'lexicon': 0.5,
                'pos_words': 0,
                'neg_words': 0,
                'negation_count': 0
            }

    def analyze_emotional_features_v2(self, text):
        """改进的情感特征分析"""
        words = jieba.lcut(text)
        total_words = max(len(words), 1)
        total_chars = max(len(text), 1)

        features = {
            'emotional_intensity': 0.0,  # 情感强度（0-1）
            'emotional_complexity': 0.0,  # 情感复杂性（0-1）
            'artistic_engagement': 0.0,  # 艺术参与度（0-1）
            'critical_depth': 0.0,  # 批判深度（0-1）
            'experimental_level': 0.0,  # 实验性（0-1）
            'political_engagement': 0.0,  # 政治参与度（0-1）
            'aesthetic_value': 0.0  # 美学价值（新增）
        }

        # 计算各特征词的数量
        feature_counts = {}
        for feature_type in self.sentiment_lexicon.keys():
            count = sum(1 for w in words if w in self.sentiment_lexicon[feature_type])
            feature_counts[feature_type] = count

        # 情感强度：基于情感词密度和多样性
        emotion_words = feature_counts.get('positive', 0) + feature_counts.get('negative', 0)
        features['emotional_intensity'] = min(1.0, emotion_words / (total_words / 10))

        # 情感复杂性：基于正负情感共存程度
        if emotion_words > 0:
            pos_ratio = feature_counts.get('positive', 0) / emotion_words
            neg_ratio = feature_counts.get('negative', 0) / emotion_words

            # 使用信息熵计算复杂性
            complexity = 0.0
            if pos_ratio > 0:
                complexity -= pos_ratio * np.log2(pos_ratio + 1e-10)
            if neg_ratio > 0:
                complexity -= neg_ratio * np.log2(neg_ratio + 1e-10)

            features['emotional_complexity'] = min(1.0, complexity)

        # 其他特征：基于特征词密度
        feature_mapping = {
            'experimental': 'experimental_level',
            'critical': 'critical_depth',
            'interactive': 'artistic_engagement',
            'political': 'political_engagement',
            'aesthetic': 'aesthetic_value'
        }

        for source_feature, target_feature in feature_mapping.items():
            count = feature_counts.get(source_feature, 0)
            # 使用对数变换避免过大值
            features[target_feature] = min(1.0, np.log1p(count) / np.log1p(total_words / 50))

        # 四舍五入到3位小数
        for key in features:
            features[key] = round(features[key], 3)

        return features

    def analyze_festival_sentiment_v2(self, festival_data):
        """改进的艺术节情感分析"""
        title = festival_data['title']
        content = festival_data['content']

        clean_content = self.preprocessor.clean_text(content)

        if len(clean_content) < 100:
            return self._get_default_result(title, clean_content)

        # 情感分析
        sentiment_scores = self.calculate_sentiment_score_v2(clean_content)

        # 情感特征分析
        emotional_features = self.analyze_emotional_features_v2(clean_content)

        # 提取元数据
        metadata = self.preprocessor.extract_metadata_v2(content)

        # 情感分类（更细致的分类）
        overall_score = sentiment_scores['overall']

        if overall_score > 0.7:
            sentiment_label = "强烈积极"
            sentiment_category = "高度正向体验"
        elif overall_score > 0.55:
            sentiment_label = "温和积极"
            sentiment_category = "正向体验"
        elif overall_score > 0.45:
            sentiment_label = "中性"
            if emotional_features['critical_depth'] > 0.3:
                sentiment_category = "批判性记录"
            else:
                sentiment_category = "客观记录"
        elif overall_score > 0.3:
            sentiment_label = "温和消极"
            sentiment_category = "反思性体验"
        else:
            sentiment_label = "强烈消极"
            sentiment_category = "批判性反思"

        # 艺术风格标签
        style_tags = self._extract_style_tags(emotional_features)

        return {
            'festival_name': title[:100],
            'text': clean_content,
            'length': len(clean_content),
            'word_count': len(jieba.lcut(clean_content)),
            'sentiment_scores': sentiment_scores,
            'emotional_features': emotional_features,
            'sentiment_label': sentiment_label,
            'sentiment_category': sentiment_category,
            'style_tags': style_tags,
            'metadata': metadata
        }

    def _extract_style_tags(self, features):
        """提取艺术风格标签"""
        tags = []

        if features['experimental_level'] > 0.6:
            tags.append("实验性")
        if features['critical_depth'] > 0.6:
            tags.append("批判性")
        if features['political_engagement'] > 0.6:
            tags.append("政治性")
        if features['aesthetic_value'] > 0.6:
            tags.append("美学性")
        if features['artistic_engagement'] > 0.6:
            tags.append("参与性")
        if features['emotional_complexity'] > 0.7:
            tags.append("情感复杂")

        return tags

    def _get_default_result(self, title, text):
        """获取默认结果（用于短文本）"""
        return {
            'festival_name': title[:100],
            'text': text,
            'length': len(text),
            'word_count': len(jieba.lcut(text)),
            'sentiment_scores': {
                'overall': 0.5,
                'snownlp': 0.5,
                'lexicon': 0.5,
                'pos_words': 0,
                'neg_words': 0,
                'negation_count': 0
            },
            'emotional_features': {
                'emotional_intensity': 0.0,
                'emotional_complexity': 0.0,
                'artistic_engagement': 0.0,
                'critical_depth': 0.0,
                'experimental_level': 0.0,
                'political_engagement': 0.0,
                'aesthetic_value': 0.0
            },
            'sentiment_label': "中性",
            'sentiment_category': "文本过短",
            'style_tags': ["简短"],
            'metadata': {'years': [], 'locations': [], 'artists': [], 'works': [], 'keywords': []}
        }


# ==================== 4. 高级分析功能（优化版） ====================
class AdvancedArtAnalyzerV2:
    """高级艺术文本分析器（优化版）"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = ArtTextSentimentAnalyzerV2()
        self.setup_visualization_style()

    def setup_visualization_style(self):
        """设置可视化样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.colors = {
            'positive': '#2E86AB',
            'negative': '#A23B72',
            'neutral': '#F18F01',
            'experimental': '#73AB84',
            'critical': '#C44900',
            'political': '#6A0572',
            'aesthetic': '#FF6B6B'
        }

    def perform_comprehensive_analysis(self, analysis_results, output_dir='情感分析结果'):
        """执行综合分析"""
        print("\n" + "=" * 80)
        print("开始高级综合分析")
        print("=" * 80)

        # 1. 情感分布分析
        print("\n1. 情感分布分析")
        sentiment_stats = self.analyze_sentiment_distribution(analysis_results)
        self.plot_sentiment_distribution(sentiment_stats, output_dir)

        # 2. 时间趋势分析
        print("\n2. 时间趋势分析")
        temporal_results = self.analyze_temporal_trends(analysis_results)
        if temporal_results:
            self.plot_temporal_trends(temporal_results, output_dir)

        # 3. 艺术特征分析
        print("\n3. 艺术特征分析")
        feature_correlations = self.analyze_feature_correlations(analysis_results)
        self.plot_feature_analysis(analysis_results, feature_correlations, output_dir)

        # 4. 文本内容分析
        print("\n4. 文本内容分析")
        content_analysis = self.analyze_text_content(analysis_results)
        self.plot_content_analysis(content_analysis, output_dir)

        # 5. 生成综合报告
        print("\n5. 生成综合报告")
        self.generate_comprehensive_report(analysis_results, output_dir)

        print("\n✓ 高级分析完成！")

    def analyze_sentiment_distribution(self, results):
        """分析情感分布"""
        sentiments = [r['sentiment_scores']['overall'] for r in results]

        stats_dict = {
            'mean': np.mean(sentiments),
            'median': np.median(sentiments),
            'std': np.std(sentiments),
            'min': np.min(sentiments),
            'max': np.max(sentiments),
            'skewness': scipy_stats.skew(sentiments) if len(sentiments) > 1 else 0,
            'kurtosis': scipy_stats.kurtosis(sentiments) if len(sentiments) > 1 else 0,
            'positive_count': sum(1 for s in sentiments if s > 0.55),
            'negative_count': sum(1 for s in sentiments if s < 0.45),
            'neutral_count': sum(1 for s in sentiments if 0.45 <= s <= 0.55),
            'sentiments': sentiments  # 保存sentiments用于绘图
        }

        # 情感标签统计
        labels = [r['sentiment_label'] for r in results]
        label_counts = Counter(labels)
        stats_dict['label_distribution'] = dict(label_counts)

        # 情感类别统计
        categories = [r['sentiment_category'] for r in results]
        category_counts = Counter(categories)
        stats_dict['category_distribution'] = dict(category_counts)

        return stats_dict

    def plot_sentiment_distribution(self, stats, output_dir):
        """绘制情感分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('情感分布综合分析', fontsize=16, fontweight='bold')

        # 1. 情感得分分布直方图
        if 'sentiments' in stats and len(stats['sentiments']) > 0:
            sentiments = stats['sentiments']
            axes[0, 0].hist(sentiments, bins=20, alpha=0.7,
                            color=self.colors['positive'], edgecolor='black')
            axes[0, 0].set_title('情感得分分布', fontsize=12)
            axes[0, 0].set_xlabel('情感得分')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=12)
            axes[0, 0].axis('off')

        # 2. 情感标签分布
        if 'label_distribution' in stats:
            labels = list(stats['label_distribution'].keys())
            counts = list(stats['label_distribution'].values())

            colors = []
            for label in labels:
                if '积极' in label:
                    colors.append(self.colors['positive'])
                elif '消极' in label:
                    colors.append(self.colors['negative'])
                else:
                    colors.append(self.colors['neutral'])

            axes[0, 1].bar(labels, counts, color=colors, alpha=0.7)
            axes[0, 1].set_title('情感标签分布', fontsize=12)
            axes[0, 1].set_xlabel('情感标签')
            axes[0, 1].set_ylabel('数量')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. 统计指标
        metrics = ['mean', 'median', 'std', 'min', 'max']
        metric_names = ['均值', '中位数', '标准差', '最小值', '最大值']
        metric_values = [stats[m] for m in metrics]

        x_pos = np.arange(len(metrics))
        axes[1, 0].bar(x_pos, metric_values, alpha=0.7, color=self.colors['experimental'])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(metric_names, fontsize=10)
        axes[1, 0].set_title('情感统计指标', fontsize=12)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, v in enumerate(metric_values):
            axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. 情感类别分布
        if 'category_distribution' in stats:
            categories = list(stats['category_distribution'].keys())
            cat_counts = list(stats['category_distribution'].values())

            axes[1, 1].pie(cat_counts, labels=categories, autopct='%1.1f%%',
                           startangle=90, colors=[self.colors['positive'],
                                                  self.colors['negative'],
                                                  self.colors['neutral']],
                           textprops={'fontsize': 9})
            axes[1, 1].set_title('情感类别分布', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 情感分布图已保存: {output_dir}/sentiment_distribution.png")

    def analyze_temporal_trends(self, results):
        """分析时间趋势"""
        temporal_data = []

        for result in results:
            # 尝试提取年份
            years = result['metadata']['years']
            year = None

            if years:
                try:
                    year = int(max(years))  # 取最大年份
                except:
                    pass

            if year:
                temporal_data.append({
                    'year': year,
                    'sentiment': result['sentiment_scores']['overall'],
                    'intensity': result['emotional_features']['emotional_intensity'],
                    'experimental': result['emotional_features']['experimental_level'],
                    'critical': result['emotional_features']['critical_depth'],
                    'political': result['emotional_features']['political_engagement'],
                    'name': result['festival_name']
                })

        if len(temporal_data) < 3:
            print("  数据不足，跳过时间趋势分析")
            return None

        df = pd.DataFrame(temporal_data)
        df = df.sort_values('year')

        # 计算年度统计
        yearly_stats = df.groupby('year').agg({
            'sentiment': ['mean', 'std', 'count'],
            'intensity': 'mean',
            'experimental': 'mean',
            'critical': 'mean'
        }).round(3)

        return {
            'temporal_df': df,
            'yearly_stats': yearly_stats,
            'trend_sentiment': np.polyfit(df['year'], df['sentiment'], 1)[0],
            'trend_experimental': np.polyfit(df['year'], df['experimental'], 1)[0]
        }

    def plot_temporal_trends(self, temporal_results, output_dir):
        """绘制时间趋势图"""
        df = temporal_results['temporal_df']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('时间趋势分析', fontsize=16, fontweight='bold')

        # 1. 情感得分随时间变化
        axes[0, 0].scatter(df['year'], df['sentiment'], alpha=0.6, s=50)

        # 添加趋势线
        z = np.polyfit(df['year'], df['sentiment'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df['year'], p(df['year']), "r--", alpha=0.7, linewidth=2)

        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('情感得分')
        axes[0, 0].set_title(f'情感趋势 (斜率: {z[0]:.4f})')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 实验性随时间变化
        axes[0, 1].scatter(df['year'], df['experimental'], alpha=0.6, s=50, color=self.colors['experimental'])

        z_exp = np.polyfit(df['year'], df['experimental'], 1)
        p_exp = np.poly1d(z_exp)
        axes[0, 1].plot(df['year'], p_exp(df['year']), "b--", alpha=0.7, linewidth=2)

        axes[0, 1].set_xlabel('年份')
        axes[0, 1].set_ylabel('实验性')
        axes[0, 1].set_title(f'实验性趋势 (斜率: {z_exp[0]:.4f})')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 特征关系散点图
        scatter = axes[1, 0].scatter(df['experimental'], df['critical'],
                                     c=df['sentiment'], cmap='RdYlGn',
                                     s=df['intensity'] * 100, alpha=0.7,
                                     edgecolors='black')

        axes[1, 0].set_xlabel('实验性')
        axes[1, 0].set_ylabel('批判深度')
        axes[1, 0].set_title('实验性 vs 批判深度 (颜色=情感, 大小=强度)')
        plt.colorbar(scatter, ax=axes[1, 0])

        # 4. 年度特征热图
        if len(df['year'].unique()) > 3:
            try:
                pivot_table = df.pivot_table(index='year',
                                             values=['sentiment', 'experimental', 'critical'],
                                             aggfunc='mean')

                im = axes[1, 1].imshow(pivot_table.T, aspect='auto', cmap='YlOrRd')
                axes[1, 1].set_xlabel('年份')
                axes[1, 1].set_yticks(range(len(pivot_table.columns)))
                axes[1, 1].set_yticklabels(['情感', '实验性', '批判深度'])
                axes[1, 1].set_title('年度特征热图')
                plt.colorbar(im, ax=axes[1, 1])
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'热图生成失败\n{str(e)[:50]}', 
                               ha='center', va='center', fontsize=10)
                axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, '数据不足，无法生成热图', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 时间趋势图已保存: {output_dir}/temporal_trends.png")

    def analyze_feature_correlations(self, results):
        """分析特征相关性"""
        # 提取特征数据
        features_data = []

        for result in results:
            features = result['emotional_features']
            features_data.append({
                'sentiment': result['sentiment_scores']['overall'],
                'intensity': features['emotional_intensity'],
                'complexity': features['emotional_complexity'],
                'engagement': features['artistic_engagement'],
                'critical': features['critical_depth'],
                'experimental': features['experimental_level'],
                'political': features['political_engagement'],
                'aesthetic': features['aesthetic_value']
            })

        df = pd.DataFrame(features_data)

        # 计算相关系数矩阵
        corr_matrix = df.corr().round(3)

        # 找出强相关关系
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr
                    })

        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'feature_data': df
        }

    def plot_feature_analysis(self, results, feature_correlations, output_dir):
        """绘制特征分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('艺术特征分析', fontsize=16, fontweight='bold')

        df = feature_correlations['feature_data']
        corr_matrix = feature_correlations['correlation_matrix']

        # 1. 特征相关性热图
        im = axes[0, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 0].set_xticks(range(len(corr_matrix.columns)))
        axes[0, 0].set_yticks(range(len(corr_matrix.columns)))
        axes[0, 0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_yticklabels(corr_matrix.columns, fontsize=9)
        axes[0, 0].set_title('特征相关性热图')

        # 添加相关系数文本
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[0, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center",
                                       color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                                       fontsize=8)

        plt.colorbar(im, ax=axes[0, 0])

        # 2. 特征分布箱线图
        feature_columns = ['intensity', 'complexity', 'engagement', 'critical', 'experimental']
        feature_names = ['情感强度', '情感复杂性', '艺术参与度', '批判深度', '实验性']

        feature_data_to_plot = [df[col] for col in feature_columns]

        bp = axes[0, 1].boxplot(feature_data_to_plot, labels=feature_names,
                                patch_artist=True)

        # 设置箱线图颜色
        colors = [self.colors['positive'], self.colors['neutral'],
                  self.colors['experimental'], self.colors['critical'],
                  self.colors['political']]

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[0, 1].set_title('特征值分布箱线图')
        axes[0, 1].set_ylabel('特征值')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. 特征雷达图（平均特征）
        avg_features = df.mean()

        # 选择要显示的6个特征
        radar_features = ['intensity', 'complexity', 'engagement', 'critical', 'experimental', 'aesthetic']
        radar_values = [avg_features[feat] for feat in radar_features]
        radar_labels = ['情感强度', '情感复杂性', '艺术参与度', '批判深度', '实验性', '美学价值']

        # 闭合雷达图
        radar_values += radar_values[:1]
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(2, 2, (3, 4), polar=True)
        ax.plot(angles, radar_values, 'o-', linewidth=2)
        ax.fill(angles, radar_values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('平均特征雷达图', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 特征分析图已保存: {output_dir}/feature_analysis.png")

        # 打印强相关关系
        if feature_correlations['strong_correlations']:
            print("\n强相关关系 (>0.5):")
            for corr in feature_correlations['strong_correlations']:
                print(f"  {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")

    def analyze_text_content(self, results):
        """分析文本内容"""
        all_text = ' '.join([r['text'] for r in results])

        # 分词
        words = jieba.lcut(all_text)
        words = [w for w in words if len(w) > 1 and w not in self.preprocessor.stopwords]

        # 词频统计
        word_freq = Counter(words)
        top_words = word_freq.most_common(50)

        # 提取高频名词、动词、形容词
        pos_words = pseg.cut(all_text)
        nouns = []
        verbs = []
        adjectives = []

        for word, flag in pos_words:
            if len(word) > 1 and word not in self.preprocessor.stopwords:
                if flag.startswith('n'):  # 名词
                    nouns.append(word)
                elif flag.startswith('v'):  # 动词
                    verbs.append(word)
                elif flag.startswith('a'):  # 形容词
                    adjectives.append(word)

        return {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'top_words': top_words[:20],
            'nouns': Counter(nouns).most_common(20),
            'verbs': Counter(verbs).most_common(20),
            'adjectives': Counter(adjectives).most_common(20),
            'lexical_diversity': len(set(words)) / max(len(words), 1)
        }

    def plot_content_analysis(self, content_analysis, output_dir):
        """绘制内容分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('文本内容分析', fontsize=16, fontweight='bold')

        # 1. 高频词汇条形图
        top_words = content_analysis['top_words'][:15]
        words = [w[0] for w in top_words]
        freqs = [w[1] for w in top_words]

        bars = axes[0, 0].barh(words[::-1], freqs[::-1], alpha=0.7, color=self.colors['positive'])
        axes[0, 0].set_xlabel('频次')
        axes[0, 0].set_title('高频词汇Top 15')

        # 2. 词性分布
        pos_data = {
            '名词': len(content_analysis['nouns']),
            '动词': len(content_analysis['verbs']),
            '形容词': len(content_analysis['adjectives'])
        }

        axes[0, 1].pie(pos_data.values(), labels=pos_data.keys(), autopct='%1.1f%%',
                       colors=[self.colors['positive'], self.colors['experimental'], self.colors['critical']],
                       startangle=90)
        axes[0, 1].set_title('词性分布')

        # 3. 生成词云
        try:
            font_path = find_chinese_font()
            wordcloud_params = {
                'width': 800,
                'height': 400,
                'background_color': 'white',
                'max_words': 100,
                'max_font_size': 100,
                'random_state': 42
            }
            if font_path:
                wordcloud_params['font_path'] = font_path
            
            wordcloud = WordCloud(**wordcloud_params).generate(' '.join([w[0] for w in content_analysis['top_words']]))

            axes[1, 0].imshow(wordcloud, interpolation='bilinear')
            axes[1, 0].axis('off')
            axes[1, 0].set_title('词云图')
        except:
            axes[1, 0].text(0.5, 0.5, '词云生成失败',
                            ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')

        # 4. 文本统计信息
        stats_text = f"""
        总词汇数: {content_analysis['total_words']:,}
        唯一词汇数: {content_analysis['unique_words']:,}
        词汇多样性: {content_analysis['lexical_diversity']:.3f}

        高频名词示例:
        {', '.join([w[0] for w in content_analysis['nouns'][:5]])}

        高频动词示例:
        {', '.join([w[0] for w in content_analysis['verbs'][:5]])}

        高频形容词示例:
        {', '.join([w[0] for w in content_analysis['adjectives'][:5]])}
        """

        axes[1, 1].text(0, 1, stats_text, fontsize=10, va='top',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('文本统计')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/content_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 内容分析图已保存: {output_dir}/content_analysis.png")

        # 保存词频数据
        word_freq_df = pd.DataFrame(content_analysis['top_words'], columns=['词汇', '频次'])
        word_freq_df.to_csv(f'{output_dir}/word_frequency.csv',
                            index=False, encoding='utf-8-sig')
        print(f"✓ 词频数据已保存: {output_dir}/word_frequency.csv")

    def generate_comprehensive_report(self, results, output_dir):
        """生成综合分析报告"""
        report_path = f'{output_dir}/comprehensive_analysis_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 行为艺术文本综合分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 总体统计
            total_festivals = len(results)
            avg_sentiment = np.mean([r['sentiment_scores']['overall'] for r in results])
            avg_length = np.mean([r['length'] for r in results])

            f.write("## 1. 总体统计\n\n")
            f.write(f"- 分析艺术节数量: {total_festivals}\n")
            f.write(f"- 平均情感得分: {avg_sentiment:.3f}\n")
            f.write(f"- 平均文本长度: {avg_length:.0f} 字符\n\n")

            # 情感分布
            f.write("## 2. 情感分布\n\n")
            sentiments = [r['sentiment_scores']['overall'] for r in results]
            labels = [r['sentiment_label'] for r in results]

            label_counts = Counter(labels)
            for label, count in label_counts.items():
                percentage = count / total_festivals * 100
                f.write(f"- {label}: {count} ({percentage:.1f}%)\n")

            f.write(f"\n情感得分范围: {min(sentiments):.3f} - {max(sentiments):.3f}\n")
            f.write(f"情感得分标准差: {np.std(sentiments):.3f}\n\n")

            # 艺术特征统计
            f.write("## 3. 艺术特征统计\n\n")
            features = ['emotional_intensity', 'experimental_level',
                        'critical_depth', 'artistic_engagement',
                        'political_engagement', 'aesthetic_value']

            feature_names = {
                'emotional_intensity': '情感强度',
                'experimental_level': '实验性',
                'critical_depth': '批判深度',
                'artistic_engagement': '艺术参与度',
                'political_engagement': '政治参与度',
                'aesthetic_value': '美学价值'
            }

            for feature in features:
                values = [r['emotional_features'][feature] for r in results]
                f.write(f"- {feature_names[feature]}: {np.mean(values):.3f} "
                        f"(范围: {min(values):.3f} - {max(values):.3f})\n")

            f.write("\n")

            # 详细分析表格
            f.write("## 4. 详细分析结果\n\n")
            f.write("| 序号 | 艺术节名称 | 情感得分 | 情感标签 | 文本长度 | 实验性 | 批判深度 |\n")
            f.write("|------|------------|----------|----------|----------|--------|----------|\n")

            for i, result in enumerate(results, 1):
                f.write(f"| {i} | {result['festival_name'][:20]} | "
                        f"{result['sentiment_scores']['overall']:.3f} | "
                        f"{result['sentiment_label']} | "
                        f"{result['length']:,} | "
                        f"{result['emotional_features']['experimental_level']:.3f} | "
                        f"{result['emotional_features']['critical_depth']:.3f} |\n")

            # 结论与发现
            f.write("\n## 5. 主要发现\n\n")

            # 计算各种比例
            high_experimental = sum(1 for r in results
                                    if r['emotional_features']['experimental_level'] > 0.6)
            high_critical = sum(1 for r in results
                                if r['emotional_features']['critical_depth'] > 0.6)
            high_political = sum(1 for r in results
                                 if r['emotional_features']['political_engagement'] > 0.6)

            f.write(f"1. **实验性特征**: {high_experimental}个艺术节具有高实验性 "
                    f"({high_experimental / total_festivals * 100:.1f}%)\n")
            f.write(f"2. **批判性特征**: {high_critical}个艺术节具有高批判深度 "
                    f"({high_critical / total_festivals * 100:.1f}%)\n")
            f.write(f"3. **政治参与**: {high_political}个艺术节具有高政治参与度 "
                    f"({high_political / total_festivals * 100:.1f}%)\n\n")

            # 建议
            f.write("## 6. 建议\n\n")
            f.write("1. 对于情感得分较低的艺术节，可以关注其批判深度和实验性价值\n")
            f.write("2. 高实验性的艺术节通常具有较高的创新价值\n")
            f.write("3. 批判性强的艺术节可能对社会议题有更深入的探讨\n")
            f.write("4. 建议结合具体文本内容进行更深入的质性分析\n")

        print(f"✓ 综合分析报告已保存: {report_path}")


# ==================== 5. 主分析流程（优化版） ====================
def analyze_art_performance_text_v2(file_path, advanced_analysis=True):
    """主分析函数（优化版）"""

    print("=" * 80)
    print("行为艺术文本深度情感分析系统 V2.0")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 错误: 找不到文件 {file_path}")
        print("请检查文件路径是否正确")
        return None, None

    # 1. 读取文本
    try:
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
        text = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"✓ 成功读取文件 (编码: {encoding})")
                print(f"✓ 文本总长度: {len(text):,} 字符")
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            print("✗ 无法读取文件，请检查文件编码")
            return None, None

    except Exception as e:
        print(f"✗ 读取文件时出错: {e}")
        return None, None

    # 2. 初始化分析器
    preprocessor = TextPreprocessor()
    analyzer = ArtTextSentimentAnalyzerV2()
    advanced_analyzer = AdvancedArtAnalyzerV2()

    # 3. 分割艺术节
    print("\n正在分割艺术节文本...")
    festivals = preprocessor.split_by_festival_v2(text)
    print(f"✓ 识别到 {len(festivals)} 个艺术节/事件")

    # 显示前几个艺术节的信息
    for i, festival in enumerate(festivals[:5], 1):
        title = festival['title']
        preview = festival['content'][:100].replace('\n', ' ').strip()
        print(f"  艺术节{i}: {title[:40]}... - {preview}...")

    # 4. 分析每个艺术节
    print(f"\n开始分析各艺术节情感...")
    analysis_results = []

    for i, festival in enumerate(festivals, 1):
        title = festival['title']
        content = festival['content']

        if len(content.strip()) < 100:
            print(f"  跳过第 {i:2d} 个: {title[:30]}... (文本过短)")
            continue

        print(f"  分析第 {i:2d} 个: {title[:40]}...")

        try:
            result = analyzer.analyze_festival_sentiment_v2(festival)
            analysis_results.append(result)

            # 显示简要结果
            sentiment = result['sentiment_scores']['overall']
            label = result['sentiment_label']
            print(f"    → 情感: {sentiment:.3f} ({label})")

        except Exception as e:
            print(f"    ✗ 分析失败: {e}")
            continue

    if not analysis_results:
        print("✗ 没有成功分析任何艺术节")
        return None, None

    print(f"✓ 成功分析 {len(analysis_results)} 个艺术节")

    # 5. 创建DataFrame
    print("\n正在生成分析结果...")
    df_data = []

    for i, result in enumerate(analysis_results, 1):
        df_data.append({
            '序号': i,
            '艺术节名称': result['festival_name'][:50],
            '情感得分': result['sentiment_scores']['overall'],
            'SnowNLP得分': result['sentiment_scores']['snownlp'],
            '词典得分': result['sentiment_scores']['lexicon'],
            '积极词数': result['sentiment_scores']['pos_words'],
            '消极词数': result['sentiment_scores']['neg_words'],
            '情感标签': result['sentiment_label'],
            '情感类别': result['sentiment_category'],
            '风格标签': ', '.join(result['style_tags']),
            '文本长度': result['length'],
            '词汇数量': result['word_count'],
            '情感强度': result['emotional_features']['emotional_intensity'],
            '情感复杂性': result['emotional_features']['emotional_complexity'],
            '艺术参与度': result['emotional_features']['artistic_engagement'],
            '批判深度': result['emotional_features']['critical_depth'],
            '实验性': result['emotional_features']['experimental_level'],
            '政治参与度': result['emotional_features']['political_engagement'],
            '美学价值': result['emotional_features']['aesthetic_value'],
            '年份': ', '.join(result['metadata']['years'][:3]),
            '地点': ', '.join(result['metadata']['locations'][:3]),
            '艺术家': ', '.join(result['metadata']['artists'][:3]),
            '关键词': ', '.join(result['metadata']['keywords'][:5])
        })

    df = pd.DataFrame(df_data)

    # 6. 创建输出目录
    output_dir = '情感分析结果_V2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 7. 保存基础结果
    try:
        # 保存CSV
        df.to_csv(f'{output_dir}/sentiment_analysis_results.csv',
                  index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: {output_dir}/sentiment_analysis_results.csv")

        # 保存Excel（包含多个sheet）
        with pd.ExcelWriter(f'{output_dir}/analysis_results.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='情感分析', index=False)

            # 添加统计摘要
            summary_data = {
                '指标': ['艺术节数量', '平均情感得分', '情感得分标准差',
                         '平均文本长度', '平均实验性', '平均批判深度'],
                '数值': [
                    len(df), 
                    round(df['情感得分'].mean(), 3), 
                    round(df['情感得分'].std(), 3),
                    round(df['文本长度'].mean(), 0), 
                    round(df['实验性'].mean(), 3), 
                    round(df['批判深度'].mean(), 3)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)

        print(f"✓ 已保存: {output_dir}/analysis_results.xlsx")

    except Exception as e:
        print(f"✗ 保存文件时出错: {e}")

    # 8. 执行高级分析
    if advanced_analysis and len(analysis_results) >= 3:
        try:
            advanced_analyzer.perform_comprehensive_analysis(analysis_results, output_dir)
        except Exception as e:
            print(f"✗ 高级分析出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n数据不足({len(analysis_results)}个)，跳过高级分析")

    # 9. 显示简要统计
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    print(f"\n简要统计:")
    print(f"- 分析艺术节数: {len(df)}")
    print(f"- 情感得分: {df['情感得分'].mean():.3f} ± {df['情感得分'].std():.3f}")
    print(f"- 得分范围: {df['情感得分'].min():.3f} - {df['情感得分'].max():.3f}")
    print(f"- 情感标签分布: {dict(df['情感标签'].value_counts())}")
    print(f"- 平均实验性: {df['实验性'].mean():.3f}")
    print(f"- 平均批判深度: {df['批判深度'].mean():.3f}")

    print(f"\n结果文件保存在: {output_dir}/")
    print("主要文件:")
    print(f"1. {output_dir}/sentiment_analysis_results.csv - 详细分析数据")
    print(f"2. {output_dir}/analysis_results.xlsx - Excel格式结果")
    print(f"3. {output_dir}/comprehensive_analysis_report.md - 综合分析报告")
    print(f"4. {output_dir}/*.png - 各种分析图表")

    return df, analysis_results


# ==================== 6. 运行分析 ====================
if __name__ == "__main__":
    # 设置文件路径 - 请修改为实际路径
    file_path = r"D:\桌面\行为艺术现场_完整全文.txt"

    print("正在启动行为艺术文本情感分析系统 V2.0...")
    print(f"文件路径: {file_path}")
    print("-" * 80)

    try:
        # 运行优化版分析
        df, results = analyze_art_performance_text_v2(file_path, advanced_analysis=True)

        if df is not None:
            print("\n数据预览:")
            print("-" * 80)
            print(df[['序号', '艺术节名称', '情感得分', '情感标签', '实验性', '批判深度']].head(10))

    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

        print("\n请检查以下可能的问题:")
        print("1. 文件路径是否正确")
        print("2. 是否安装了所有依赖库")
        print("3. 文件编码是否为UTF-8")
        print("\n依赖库安装命令:")
        print("pip install jieba snownlp pandas matplotlib seaborn wordcloud scipy scikit-learn openpyxl")