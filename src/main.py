import requests
from bs4 import BeautifulSoup
import numpy as np
import re

# clean = lambda s: str(re.sub('[\W_]+', ' ', s))


def get_text(article_url):
    res = requests.get(article_url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    article = soup.body.findAll('article')
    # text = ' '.join([clean(s.text) for s in article[0].findAll('p')])
    text = ' '.join([s.text for s in article[0].findAll('p')])
    return text


text = get_text("https://www.washingtonpost.com/national-security/2022/05/21/russia-ukraine-victory/")
print(text)

# list_of_articles = article_urls(blog_address)
# print(list_of_articles)

# blogs = {}
# for article in list_of_articles:
#     text = get_text(article)
#     blogs[article] = text

# f = open('blogs.txt', "wb")
# for i in blogs:
#     f.write(i + '\t' + blogs[i] + '\n')
# f.close()

# url = 'https://www.washingtonpost.com/national-security/2022/05/21/russia-ukraine-victory/'
# res = requests.get(url)
# html_page = res.content
# soup = BeautifulSoup(html_page, 'html.parser')
# text = soup.find_all(text=True)
#
# output = ''
# blacklist = [
#     '[document]',
#     'noscript',
#     'header',
#     'html',
#     'meta',
#     'head',
#     'input',
#     'script',
#     # there may be more elements you don't want, such as "style", etc.
# ]
#
# for t in text:
#     if t.parent.name not in blacklist:
#         output += '{} '.format(t)
#
# print(output)
