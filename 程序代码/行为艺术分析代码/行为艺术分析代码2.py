import sys
import time

# 检查必要的库是否已安装
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    print("缺少必要的库，请先安装：")
    print("pip install requests beautifulsoup4 lxml")
    sys.exit(1)


def scrape_wikipedia_simple(url):
    """
    简化版本的维基百科爬虫，不使用正则表达式
    """
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        print("正在发送请求...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        print("请求成功，正在解析页面...")

        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取标题
        title_element = soup.find('h1', class_='firstHeading')
        title = title_element.get_text().strip() if title_element else "未知标题"

        # 提取主要内容
        content_div = soup.find('div', id='mw-content-text')

        paragraphs = []
        if content_div:
            # 获取所有段落
            p_tags = content_div.find_all('p')
            for p in p_tags:
                text = p.get_text().strip()
                # 使用简单的字符串替换代替正则表达式
                text = text.replace('\n', ' ').replace('\t', ' ')
                # 合并多个空格
                while '  ' in text:
                    text = text.replace('  ', ' ')

                if len(text) > 20:  # 只保留有意义的段落
                    paragraphs.append(text)

        # 构建结果
        result = {
            'title': title,
            'url': url,
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs),
            'first_paragraph': paragraphs[0] if paragraphs else "无内容"
        }

        return result

    except Exception as e:
        print(f"错误: {e}")
        return None


# 主程序
if __name__ == "__main__":
    target_url = "https://zh.wikipedia.org/wiki/%E8%A1%8C%E4%B8%BA%E8%89%BA%E6%9C%AF"

    print(f"开始爬取: {target_url}")

    # 爬取数据
    data = scrape_wikipedia_simple(target_url)

    if data:
        print("✓ 爬取成功！")
        print(f"页面标题: {data['title']}")
        print(f"获取段落数: {data['paragraph_count']}")

        # 显示第一段内容
        if data['first_paragraph']:
            print("\n第一段内容:")
            print("-" * 50)
            print(data['first_paragraph'][:300] + "..." if len(data['first_paragraph']) > 300 else data[
                'first_paragraph'])
            print("-" * 50)

        # 保存结果
        try:
            filename = f"{data['title']}_内容.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"标题: {data['title']}\n\n")
                for i, para in enumerate(data['paragraphs'], 1):
                    f.write(f"段落 {i}:\n{para}\n\n")
            print(f"✓ 内容已保存到: {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    else:
        print("❌ 爬取失败")

    print("程序执行完毕")
