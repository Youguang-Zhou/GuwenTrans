# 本脚本修改自：https://github.com/NiuTrans/Classical-Modern


import os
import random
import re
import time
from os.path import join as pathjoin
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DIR_NAME = 'data/双语数据'
LOG_FILE = 'log.txt'
BASE_URL = 'https://so.gushiwen.cn'


USER_AGENT = [
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0',
    'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)',
    'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11',
    'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11',
]


global progress_bar


def request(url):
    res = requests.get(url, headers={'User-Agent': random.choice(USER_AGENT)})
    return BeautifulSoup(res.text, 'lxml')


def save_files(save_dir, html):
    with open(os.path.join(save_dir, 'html.txt'), 'w') as f:
        f.write(str(html))


# 解析具体章节，并写入文件
def chapter(chapter_url, chapter_save_dir):
    # 请求
    bs = request(chapter_url)
    # 检查章节是否为空
    h1 = bs.select('body > div.main3 > div.left > div.sons > div.cont > h1')
    if len(h1) == 0:
        print(f'【{chapter_url}】网页内容为空。')
        return
    header = str(h1[0])
    # 第一种类型，原文和译文一一对应
    # 第二种类型，篇章级双语数据
    # 第三种类型，没有译文
    trans_url = ''
    if len(re.findall(r':ShowYizhuYuanchuang', header)) > 0:
        # 第一种类型，原文和译文一一对应
        chapter_id = re.findall(r':ShowYizhuYuanchuang\(\'(.*)\',\'duanyi\'\)', header)[0]
        trans_url = f'{BASE_URL}/guwen/ajaxbfanyiYuanchuang.aspx?id={chapter_id}&state=duanyi'
        # 请求
        bs2 = request(trans_url)
        html = bs2.select('body > div.contson')[0].decode_contents()
    elif len(re.findall(r':ShowYizhu\(', header)) > 0:
        # 第二种类型，篇章级双语数据，e.g. https://so.gushiwen.cn/guwen/bookv_abb43bda3d53.aspx
        chapter_id = re.findall(r':ShowYizhu\(\'(.*)\',\'.*\'\)', header)[0]
        trans_url = f'{BASE_URL}/guwen/ajaxbfanyi.aspx?id={chapter_id}'
        # 请求
        bs3 = request(trans_url)
        html_src = bs.select('body > div.main3 > div.left > div.sons > div.cont > div.contson')[0].decode_contents()
        html_tgt = bs3.select('body > div.sons > div.shisoncont > div.contson')[0].decode_contents()
        html = f'<section>{html_src}</section><section>{html_tgt}</section>'
    else:
        # 第三种类型，没有译文，体现为空文件夹
        return

    save_files(chapter_save_dir, html)
    time.sleep(1)


# 解析具体古籍，并按篇章和文章创建文档
def book(book_url, book_save_dir, last_info, f_log):
    # 断点续爬
    last_section = last_info[1] if isinstance(last_info, tuple) else None
    last_chapter = last_info[2] if isinstance(last_info, tuple) else None
    flag = True
    # 进度条
    global progress_bar
    postfix = progress_bar.postfix
    # 统计一共多少个篇章
    bs = request(book_url)
    sections = bs.select('body > div.main3 > div.left > div.sons > div.bookcont')
    section_num = len(sections)
    # 收集篇章名及篇章下的文章名（部分书籍没有篇章名，只有文章名）
    for i in range(section_num):
        chapters = str(sections[i])
        section_save_dir = book_save_dir

        if section_num > 1:  # 多篇章书籍
            section_name = re.findall(r'<strong>(.*)</strong>', chapters)[0]
            section_name = section_name.replace('/', '&')  # e.g. 溯江纪源 / 江源考
            if last_section is not None and last_section != section_name and flag:
                continue
            flag = False
            section_save_dir = os.path.join(section_save_dir, section_name)
            if not os.path.exists(section_save_dir):
                os.mkdir(section_save_dir)
                f_log.write('###' + section_name + '###\n')
                progress_bar.set_postfix_str(f'{postfix} 当前篇章: {section_name}')

        # [(uri, name), ...]
        chapter_uri_name = re.findall(r'<a href=\"(.*)\">(.*)</a>', chapters)

        flag2 = True
        postfix_postfix = progress_bar.postfix
        for chapter_uri, chapter_name in chapter_uri_name:
            chapter_name = chapter_name.replace('/', '&')  # e.g. 溯江纪源 / 江源考
            if last_chapter is not None and last_chapter != chapter_name and flag2:
                continue
            flag2 = False
            chapter_save_dir = os.path.join(section_save_dir, chapter_name)
            if not os.path.exists(chapter_save_dir):
                os.mkdir(chapter_save_dir)
                f_log.write('##' + chapter_name + '##\n')
                progress_bar.set_postfix_str(f'{postfix_postfix} 当前章节: {chapter_name}')
            chapter(urljoin(BASE_URL, chapter_uri), chapter_save_dir)
            last_chapter = None
            time.sleep(1)
        time.sleep(1)


# 解析具体经部、史部、子部、集部网页下的每本书
def books(part_url, books_save_dir, last_info, f_log):
    # 请求
    bs = request(part_url)
    # 提取书链接和书名
    book_list = str(bs.select('body > div.main3 > div.left > div.sons > div.typecont')[0])
    # [(uri, name), ...]
    book_uri_name = re.findall(r'<a href=\"(.*)\">(.*)</a>', book_list)
    # 断点续爬
    last_book = last_info[0] if isinstance(last_info, tuple) else None
    flag = True
    # 进度条
    global progress_bar
    progress_bar = tqdm(book_uri_name, desc=re.findall(r'type=(.*)', part_url)[0])
    for book_uri, book_name in progress_bar:
        # last_book 不为空说明是断点续爬
        if last_book is not None and book_name != last_book and flag:
            continue
        # 结束断点续爬模式/正常爬取 均是 flag = False
        flag = False
        book_save_dir = os.path.join(books_save_dir, book_name)
        if not os.path.exists(book_save_dir):
            os.mkdir(book_save_dir)
            f_log.write('####' + book_name + '####\n')
            progress_bar.set_postfix_str(f'当前书籍: {book_name}')
        book(urljoin(BASE_URL, book_uri), book_save_dir, last_info, f_log)
        last_info, last_book = None, None
        time.sleep(1)
    # 该本书没找到上个断点时返回True
    return True if flag else False


def download_books(books_main_url, books_save_dir):
    '''
    下载古籍
    '''
    # 建立保存的文件夹
    os.makedirs(books_save_dir, exist_ok=True)

    # 读取日志信息
    def parse_log(log_file):
        if not os.path.exists(log_file):
            return None
        with open(log_file) as f:
            log = f.read()
            # 最后一次爬取断点
            last_book, last_section, last_chapter = None, None, None
            # 读取最后一次爬取时最后一本书籍名称
            if len(re.findall(r'####(.*)####', log)) > 0:
                last_book = re.findall(r'####(.*)####', log)[-1]
            else:
                return None
            # 读取最后一次爬取时最后一本书的篇章名称（可能为空）
            book_content = log[log.find(last_book) :]
            if len(re.findall(r'###(.*)###', book_content)) > 0:  # 包含篇章
                last_section = re.findall(r'###(.*)###', book_content)[-1]
                book_content = book_content[book_content.find(last_section) :]
            # 读取最后一篇文章
            if len(re.findall(r'##(.*)##', book_content)) > 0:
                last_chapter = re.findall(r'##(.*)##', book_content)[-1]
            return last_book, last_section, last_chapter

    # 断点续爬，读取脚本日志文件
    log_file = os.path.join(DIR_NAME, '.cache/books_download_log.txt')
    last_info = parse_log(log_file)

    # 爬取日志
    f_log = open(log_file, 'a', buffering=1)
    for part in ['经部', '史部', '子部', '集部']:
        part_url = f'{books_main_url}/Default.aspx?p=1&type={part}'
        if not books(part_url, books_save_dir, last_info, f_log):
            last_info = None
        time.sleep(1)
    f_log.close()


def download_poetry(poetry_main_url, poetry_save_dir):
    '''
    下载古诗
    '''
    # 建立保存的文件夹
    os.makedirs(poetry_save_dir, exist_ok=True)
    # 断点续爬
    log_file = os.path.join(DIR_NAME, '.cache/poetry_download_log.txt')
    downloaded_uri_arr = []
    try:
        with open(log_file) as f:
            downloaded_uri_arr = f.read().splitlines()
    except FileNotFoundError:
        pass
    f_log = open(log_file, 'a', buffering=1)

    # 获取所有古诗的详情页uri
    all_uri_arr = []
    cache_file = os.path.join(DIR_NAME, '.cache/poetry_all_uri.txt')
    if os.path.exists(cache_file):
        # 如果已缓存，则直接读取
        with open(cache_file) as f:
            all_uri_arr = f.read().splitlines()
        print(f'【{cache_file}】读取完毕。')
    else:
        # 否则重新爬取并缓存
        bs = request(poetry_main_url)
        html = str(bs.select('body > div.main3 > div.left > div.titletype')[0])
        # 类型：... ---> tstr_arr
        # 作者：... ---> astr_arr
        # 朝代：... ---> cstr_arr
        # 形式：... ---> xstr_arr
        tstr_arr = re.findall(r'<a href=\"(/shiwens/default\.aspx\?tstr=.*?)\"', html)
        astr_arr = re.findall(r'<a href=\"(/shiwens/default\.aspx\?astr=.*?)\"', html)
        cstr_arr = re.findall(r'<a href=\"(/shiwens/default\.aspx\?cstr=.*?)\"', html)
        xstr_arr = re.findall(r'<a href=\"(/shiwens/default\.aspx\?xstr=.*?)\"', html)

        def extract_tstr_uri(tstr_arr):
            '''
            优先处理需要重定向的网页 e.g. /shiwens/default.aspx?tstr=诗经
            '''
            uri_arr = []
            for tstr in tqdm(tstr_arr, desc='古诗', leave=False):
                bs = request(url=urljoin(BASE_URL, tstr))
                redirect = re.findall(r'window\.location=\'(.*)\'', str(bs))
                if len(redirect) != 0:
                    bs = request(url=urljoin(BASE_URL, redirect[0]))
                html = str(bs.select('body > div.main3 > div.left > div.sons')[0])
                uri_arr.extend(re.findall(r'<a href=\".*(/shiwenv_.*?)\"', html))
                time.sleep(1)
            return uri_arr

        all_uri_arr = extract_tstr_uri(tstr_arr)
        for uri in tqdm([*astr_arr, *cstr_arr, *xstr_arr], desc='古诗', leave=False):
            bs = request(url=urljoin(BASE_URL, uri))
            html = str(bs.select('body > div.main3 > div.left > div.sons')[0])
            all_uri_arr.extend(re.findall(r'<a href=\".*(/shiwenv_.*?)\"', html))
            time.sleep(1)
        all_uri_arr = sorted(set(all_uri_arr))

        with open(cache_file, 'w') as f:
            for uri in all_uri_arr:
                f.write(uri + '\n')

    #################
    # 开始下载所有古诗 #
    #################
    uri_to_download = sorted(set(all_uri_arr) - set(downloaded_uri_arr))
    progress_bar = tqdm(uri_to_download, desc='古诗')
    for uri in progress_bar:

        bs = request(BASE_URL + uri)
        html = bs.select('body > div.main3 > div.left > div.sons > div.cont')
        if len(html) == 0:
            print(f'【{uri}】网页内容为空。')
            continue
        html = str(html[0])

        title = re.findall(r'<h1 .*>(.*)</h1>', html)[0]
        poetry_id = re.findall(r'OnYiwen230427\(\'.*\',\'(.*)\'\)', html)[0]
        progress_bar.set_postfix_str(f'当前古诗: {title}')

        poetry_path = os.path.join(poetry_save_dir, f'{title}_{poetry_id}')
        os.makedirs(poetry_path, exist_ok=True)

        def poetry(poetry_path, poetry_id):
            trans_url = f'{BASE_URL}/nocdn/ajaxshiwencont230427.aspx?id={poetry_id}&value=yi'
            bs = request(trans_url)
            html = bs.select('body > div.contson')[0].decode_contents()
            save_files(poetry_path, html)

        poetry(poetry_path, poetry_id)
        f_log.write(uri + '\n')
        time.sleep(1)

    f_log.close()


def main():
    # 建立缓存文件夹
    os.makedirs(os.path.join(DIR_NAME, '.cache'), exist_ok=True)
    # 下载古诗 https://so.gushiwen.cn/shiwens/
    download_poetry(
        poetry_main_url=urljoin(BASE_URL, 'shiwens'),
        poetry_save_dir=pathjoin(DIR_NAME, '所有古诗'),
    )
    # 下载古籍 (https://so.gushiwen.cn/guwen/)
    download_books(
        books_main_url=urljoin(BASE_URL, 'guwen'),
        books_save_dir=pathjoin(DIR_NAME, '所有古籍'),
    )


if __name__ == '__main__':
    main()
