# 通过 crawl.py 获取名为 “双语数据” 的文件夹后运行本脚本
# 样例输出：
#   Number of files: 17,884
#   Number of samples before deduplicate: 152,139
#   Number of samples after deduplicate: 142,672 (9,467 deleted)
#   Number of vocabulary: 13,326
#   Number of words: 66,449,654
#   First 10 vocabs:
#   ['、', '。', '々', '《', '》', '㐌', '㐨', '㑊', '㑛', '㑹']
#   Last 10 vocabs:
#   ['𤣱', '𤸷', '𥉸', '𦈡', '𧄍', '𧿒', '𩇕', '𩩲', '𩿨', '𬯎']


import glob
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

SPECIAL_CHARS = [
    ##### at begining after sorted #####
    '#',
    '$',
    '*',
    '+',
    '-',
    '/',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '=',
    '@',
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z',
    '[',
    '\\',
    ']',
    '^',
    '_',
    '`',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '{',
    '|',
    '}',
    '~',
    '\x7f',
    '¤',
    '§',
    '¨',
    '±',
    '²',
    '³',
    '·',
    '¹',
    '×',
    'à',
    'á',
    'è',
    'é',
    'ì',
    'í',
    'ò',
    'ó',
    '÷',
    'ù',
    'ú',
    'ā',
    'ē',
    'ě',
    'ī',
    'ň',
    'ō',
    'ū',
    'ǎ',
    'ǐ',
    'ǒ',
    'ǔ',
    'ǖ',
    'ǘ',
    'ǚ',
    'ǜ',
    'Γ',
    'Ε',
    'Θ',
    'Ξ',
    'Σ',
    'Υ',
    'Ψ',
    'β',
    'σ',
    'τ',
    'χ',
    'В',
    'Е',
    'Р',
    'У',
    'Х',
    'Я',
    'е',
    'л',
    'м',
    'ч',
    'ш',
    'щ',
    'ъ',
    'ы',
    'ь',
    'ю',
    '\u200b',
    '—',
    '―',
    '‖',
    '•',
    '…',
    '\u202c',
    '\u202e',
    '⁰',
    '⁴',
    '⁵',
    '⁶',
    '⁷',
    '⁸',
    '⁹',
    '℃',
    'Ⅱ',
    'ⅲ',
    '↓',
    '√',
    '∴',
    '∶',
    '①',
    '②',
    '③',
    '④',
    '⑤',
    '⑥',
    '⑦',
    '⑧',
    '⑨',
    '⑩',
    '⑹',
    '⑺',
    '⑼',
    '⑽',
    '⑿',
    '⒂',
    '⒃',
    '⒐',
    '⒑',
    '⒔',
    '⒕',
    '⒗',
    '⒙',
    '─',
    '┆',
    '┨',
    '┩',
    '┫',
    '┸',
    '┺',
    '╃',
    '■',
    '□',
    '▲',
    '△',
    '◆',
    '○',
    '◎',
    '●',
    '〈',
    '〉',
    '「',
    '」',
    '『',
    '』',
    '【',
    '】',
    '〓',
    '〔',
    '〕',
    '〖',
    '〗',
    'す',
    'で',
    'な',
    'ぶ',
    'め',
    'よ',
    'ィ',
    'ザ',
    'ド',
    'ブ',
    'ㄊ',
    'ㄌ',
    'ㄎ',
    'ㄏ',
    'ㄐ',
    'ㄗ',
    'ㄦ',
    ##### at the end after sorted #####
    '︰',
    '︴',
    '﹐',
    '﹑',
    '﹔',
    '﹕',
    '﹖',
    '＂',
    '＃',
    '＆',
    '（',
    '）',
    '＊',
    '＋',
    '－',
    '．',
    '／',
    '０',
    '１',
    '２',
    '３',
    '４',
    '５',
    '６',
    '７',
    '８',
    '９',
    '＜',
    '＝',
    '＠',
    'Ａ',
    'Ｂ',
    'Ｃ',
    'Ｄ',
    'Ｅ',
    'Ｆ',
    'Ｇ',
    'Ｈ',
    'Ｉ',
    'Ｊ',
    'Ｋ',
    'Ｌ',
    'Ｍ',
    'Ｎ',
    'Ｏ',
    'Ｐ',
    'Ｑ',
    'Ｒ',
    'Ｓ',
    'Ｔ',
    'Ｕ',
    'Ｖ',
    'Ｗ',
    'Ｘ',
    'Ｙ',
    'Ｚ',
    '［',
    '＼',
    '］',
    '＿',
    '｀',
    'ａ',
    'ｂ',
    'ｃ',
    'ｄ',
    'ｅ',
    'ｆ',
    'ｇ',
    'ｈ',
    'ｉ',
    'ｊ',
    'ｋ',
    'ｌ',
    'ｍ',
    'ｎ',
    'ｏ',
    'ｐ',
    'ｑ',
    'ｒ',
    'ｓ',
    'ｔ',
    'ｕ',
    'ｗ',
    'ｘ',
    'ｙ',
    'ｚ',
    '｛',
    '｝',
    '～',
    '￡',
    '𠏙',
    '𡸖',
    '𡽁',
    '𡾆',
    '𢓨',
    '𢕟',
    '𢤧',
    '𣬈',
    '𣵺',
    '𣶞',
    '𣸎',
    '𣺰',
    '𣽴',
    '𣾍',
    '𤄶',
    '𤞞',
    '𤟇',
    '𤟧',
    '𤡳',
    '𤢺',
    '𤽉',
    '𥌯',
    '𦁎',
    '𦍙',
    '𦓖',
    '𦛗',
    '𦶑',
    '𦶬',
    '𦸈',
    '𧁾',
    '𧈫',
    '𧮒',
    '𨲠',
    '𨴯',
    '𩇯',
    '𩘻',
    '𩩻',
    '𩭝',
    '𩳁',
    '𪂧',
    '𪃑',
    '𪄀',
    '𪄶',
    '𪆻',
    '𪇱',
    '𪓛',
    '𪓬',
    '𪓹',
]


def remove_empty(src_samples, tgt_samples):
    '''
    删除空行
    '''
    assert len(src_samples) == len(tgt_samples)
    num_samples = len(src_samples)

    src_samples_cleaned = []
    tgt_samples_cleaned = []
    for i in tqdm(range(num_samples), desc='Removing empty lines', leave=False):
        if src_samples[i] == '' or tgt_samples[i] == '':
            continue
        src_samples_cleaned.append(src_samples[i])
        tgt_samples_cleaned.append(tgt_samples[i])

    return src_samples_cleaned, tgt_samples_cleaned


def deduplicate(src_samples, tgt_samples):
    '''
    去除 src 和 tgt 完全一样的句子对
    '''
    assert len(src_samples) == len(tgt_samples)
    num_samples = len(src_samples)
    print(f'Number of samples before deduplicate: {num_samples:,}')

    num_duplicates = 0
    src_samples_cleaned = []
    tgt_samples_cleaned = []
    for i in tqdm(range(num_samples), desc='Deduplicating', leave=False):
        if src_samples[i] == tgt_samples[i]:
            num_duplicates += 1
        else:
            src_samples_cleaned.append(src_samples[i])
            tgt_samples_cleaned.append(tgt_samples[i])
    assert len(src_samples_cleaned) == len(tgt_samples_cleaned)
    num_cleaned = len(src_samples_cleaned)

    print(f'Number of samples after deduplicate: {num_cleaned:,} ({num_duplicates:,} deleted)')

    return src_samples_cleaned, tgt_samples_cleaned


def handle_special_chars(samples):
    '''
    处理特殊字符
    '''
    removed = []
    for sent in tqdm(samples, desc='Handling special characters', leave=False):
        # 移除空格
        sent = re.sub(r'\s', '', sent)
        # 移除多余html相关字符和标签
        sent = re.sub(r'&lt;', '', sent)
        sent = re.sub(r'&gt;', '', sent)
        sent = re.sub(r'</*p.*?>', '', sent)
        sent = re.sub(r'</*u.*?>', '', sent)
        sent = re.sub(r'</*strong.*?>', '', sent)
        sent = re.sub(r'</*ill.*?>', '', sent)
        sent = re.sub(r'<br/>', '', sent)
        # 替换中英文相似符号
        sent = sent.replace('!', '！')
        sent = sent.replace('(', '（')
        sent = sent.replace(')', '）')
        sent = sent.replace(',', '，')
        sent = sent.replace('.', '。')
        sent = sent.replace(':', '：')
        sent = sent.replace(';', '；')
        sent = sent.replace('?', '？')
        # 移除所有中英文引号
        sent = re.sub(r'[“”‘’"\']', '', sent)
        # 移除括号
        sent = re.sub(r'\[.*?\]', '', sent)
        sent = re.sub(r'{.*?}', '', sent)
        sent = re.sub(r'〈.*?〉', '', sent)
        sent = re.sub(r'【.*?】', '', sent)
        sent = re.sub(r'（.*?）', '', sent)
        # 移除特殊符号
        for char in SPECIAL_CHARS:
            sent = sent.replace(char, '')
        # 移除符号例如：\ue000
        sent = ''.join(filter(str.isprintable, sent))
        # 移除重复标点符号
        sent = re.sub(r'！！+', '！', sent)
        sent = re.sub(r'，，+', '，', sent)
        sent = re.sub(r'。。+', '。', sent)
        sent = re.sub(r'：：+', '：', sent)
        sent = re.sub(r'；；+', '；', sent)
        sent = re.sub(r'？？+', '？', sent)
        # 移除空书名号
        sent = sent.replace('《》', '')
        # 中文分号改成中文逗号
        sent = sent.replace('；', '，')
        # 加到数组
        removed.append(sent)
    return removed


def check_empty(src_samples, tgt_samples):
    '''
    检查是否有空行 (空行会导致梯度为nan)
    '''
    assert len(src_samples) == len(tgt_samples)
    num_samples = len(src_samples)
    for i in tqdm(range(num_samples), desc='Checking empty lines', leave=False):
        if src_samples[i] == '':
            print(src_samples[i], tgt_samples[i])
            raise Exception(f'{i}-th src 文件为空')
        if tgt_samples[i] == '':
            print(src_samples[i], tgt_samples[i])
            raise Exception(f'{i}-th tgt 文件为空')
    return src_samples, tgt_samples


def parse_html(html_files):
    '''
    获取 src 和 tgt
    '''
    src_samples = []
    tgt_samples = []
    open('data/no_trans.txt', 'w').close()
    f_no_trans = open('data/no_trans.txt', 'a', buffering=1)
    print(f'Number of files: {len(html_files):,}')
    for html_file in tqdm(html_files, desc='Parsing HTML files', leave=False):
        with open(html_file) as f:
            html = f.read()
            html = html.replace('\n', '')
            if html == '':
                # 文件为空
                continue
            if '</span>' not in html and '</section>' not in html:
                # 没有译文 e.g. https://so.gushiwen.cn/shiwenv_722b5f3aaea6.aspx
                f_no_trans.write('=' * 30 + '\n')
                f_no_trans.write(html + '\n')
                continue

            if 'section' in html:
                src, tgt = re.findall(r'<section>(.*?)</section>', html)
                src_samples.append(src)
                tgt_samples.append(tgt)
                continue

            temp = []
            for p in BeautifulSoup(html, 'html.parser').contents:
                p = str(p)
                if re.sub(r'\s', '', p) == '':
                    continue
                elif p == '<br/>':
                    pass
                elif re.match(r'^<p.*?>.*?</p>$', p):
                    pass
                elif p.endswith('</span>'):
                    p = p + '</p>'
                elif not re.match(r'^<p.*?>', p):
                    p = '<p>' + p
                temp.append(p)
            html = ''.join(temp)

            for p in BeautifulSoup(html, 'html.parser').contents:
                p = re.sub(r'\s', '', str(p))
                # src
                # 1. 查找原文
                # 2. 删除原文<br/>标签
                # 3. 删除原文<span>标签
                src = re.findall(r'<p.*?>(<br/>)?(.*?)<br/>', p)
                src = [group2 for _, group2 in src]
                src = [re.sub(r'<span.*?>.*?</span>', '', s) for s in src]
                # tgt
                # 1. 先把p中的src删掉（因为src里也可能有<span>标签）
                # 2. 查找译文<span>标签
                # 3. 删除空白<span>标签
                # 4. 合并嵌套<span>标签
                # 5. 删除多余<span>标签
                tgt = re.sub(r'<p.*?>(.*?)<br/>', '', p)
                tgt = re.findall(r'<span.*?>(.*?)(<br/>)?</span>', tgt)
                tgt = [group1 for group1, _ in tgt if group1 != '']
                tgt = [''.join(tgt)] if len(tgt) != 0 else tgt
                tgt = [re.sub(r'<span.*?>', '', t) for t in tgt]
                # 没有译文
                if len(src) == 1 and len(tgt) == 0:
                    continue
                if len(src) != len(tgt):
                    raise Exception('len(src) != len(tgt)')
                src_samples.extend(src)
                tgt_samples.extend(tgt)

    assert len(src_samples) == len(tgt_samples)
    f_no_trans.close()

    return src_samples, tgt_samples


def main():
    html_files = glob.glob('data/双语数据/**/html.txt', recursive=True)

    src_samples, tgt_samples = parse_html(html_files)

    src_samples, tgt_samples = check_empty(src_samples, tgt_samples)
    src_samples = handle_special_chars(src_samples)
    tgt_samples = handle_special_chars(tgt_samples)
    src_samples, tgt_samples = deduplicate(src_samples, tgt_samples)
    src_samples, tgt_samples = remove_empty(src_samples, tgt_samples)
    src_samples, tgt_samples = check_empty(src_samples, tgt_samples)

    vocab = []
    num_words = 0
    for i in src_samples:
        vocab.extend(list(i))
        num_words += len(list(i))
    for i in tgt_samples:
        vocab.extend(list(i))
        num_words += len(list(i))
    vocab = sorted(set(vocab))
    print(f'Number of vocabulary: {len(vocab):,}')
    print(f'Number of words: {num_words:,}')

    print(f'First 10 vocabs:\n{vocab[:10]}')
    print(f'Last 10 vocabs:\n{vocab[-10:]}')

    with open('data/src.txt', 'w') as f:
        f.writelines([s + '\n' for s in src_samples])
    with open('data/tgt.txt', 'w') as f:
        f.writelines([t + '\n' for t in tgt_samples])


if __name__ == '__main__':
    main()
