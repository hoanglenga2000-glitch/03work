from DrissionPage import ChromiumPage
from datetime import datetime
import csv
f = open('data1.csv', mode = 'w', encoding = 'utf-8', newline='')
csv_writer = csv.DictWriter(f,fieldnames = ['昵称','地区','日期','评论'])
csv_writer.writeheader()
dp = ChromiumPage()
dp.listen.start('comment/list/')
dp.get('https://www.douyin.com/video/7113959667172117790')
for page in range(1,201):
    print(f'正在采集第{page}页的数据内容')
    r = dp.listen.wait()
    json_data = r.response.body
    #print(json_data)
    comments = json_data['comments']
    for index in comments:
        c_time = index['create_time']
        date = str(datetime.fromtimestamp(c_time))
        try:
            ip_label = index['ip_label']
        except:
            ip_label = '未知'
        dit = {'昵称':index['user']['nickname'],
               '地区':index['ip_label'],
               '日期':date,
               '评论':index['text']}
        csv_writer.writerow(dit)
        print(dit)
    dp.scroll.to_see('css:.ayFW3zux')