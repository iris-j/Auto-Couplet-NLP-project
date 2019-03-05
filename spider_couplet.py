import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "keep-alive",
    "Cookie": "Hm_lvt_f6c4fb6a16d50ad6d5552e392ce43137=1547043110; Hm_lpvt_f6c4fb6a16d50ad6d5552e392ce43137=1547043426",
    "Host": "www.duiduilian.com",
    "Referer": "http://www.duiduilian.com/chunlian/4zi.html",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}
basedir = 'C:/Myprogram/Pycharm Projects/spider_xiechen'
path = '/couplet/sizi/'
if not os.path.exists(basedir + path):
    os.makedirs(basedir + path)

f1 = open(basedir + path + '/first.txt', 'w')
f2 = open(basedir + path + '/next.txt', 'w')

for i in range(1, 9):
    if i == 1:
        url = "http://www.duiduilian.com/chunlian/4zi.html"
    else:
        url = "http://www.duiduilian.com/chunlian/4zi_%s.html" % i
    html = requests.get(url, headers=headers)
    html.encoding = "utf-8"
    soup = BeautifulSoup(html.content, features="html.parser")
    block = soup.find_all(class_="content_zw")

    for j in block:
        j = re.sub(r'</?\w+[^>]*>', "", str(j))
        j = re.sub(r'（.*）', "", str(j))
        result = j.split('\n')
        result = result[1:-4]
        for k in range(len(result)):
            res = str(result[k])
            res = res.split("，")
            print(res[0], ',', res[1])
            f1.write(res[0]+'\n')
            f2.write(res[1]+'\n')

f1.close()
f2.close()

"""
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Cookie": "Hm_lvt_d86954201130d615136257dde062a503=1547296312; Hm_lpvt_d86954201130d615136257dde062a503=1547296312; doctaobaocookie=1; BDTUJIAID=335ae96c90a350250602902337d120e5",
    "Host": "www.360doc.com",
    "If-Modified-Since": "Sat, 12 Jan 2019 12:30:23 GMT",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"

}
basedir = 'C:/Myprogram/Pycharm Projects/spider_xiechen'
path = '/couplet/360/'
if not os.path.exists(basedir + path):
    os.makedirs(basedir + path)

f1 = open(basedir + path + '/first.txt', 'w')
f2 = open(basedir + path + '/next.txt', 'w')

url = "http://www.360doc.com/content/17/1208/19/1003261_711342387.shtml"
html = requests.get(url, headers=headers)
html.encoding = "utf-8"
soup = BeautifulSoup(html.content, features="html.parser")
pattern = '<p style="font-family: punctuation, 微软雅黑, Tohoma; font-size: 14px; padding: 0px; margin: 1em 0px;">'
duilian = soup.find_all(pattern)
"""