from multiprocessing.pool import ThreadPool
import urllib
import urllib.request
from bs4 import BeautifulSoup
import sys
import numpy as np
"""
python abstract_fetch.py Y.txt Y-out.txt
"""


def _get_text(query):
    try:
        page = urllib.request.urlopen(query[1])
    except urllib.error.HTTPError as e:
        return query[0], ""
    if page.info().get_content_charset() is not None:
        soup = BeautifulSoup(page.read().decode(
            page.info().get_content_charset()), 'lxml')
    else:
        return query[0], ""
    content_div = soup.find('div', {"id": "content", "class": "mw-body"})
    if content_div is not None:
        for sup in content_div.findAll('sup'):
            sup.extract()

        for a in content_div.findAll('a', {"class": "mw-jump-link"}):
            a.extract()
        if content_div.find('h1', {"class": "firstHeading"}).text.strip() == "Search results":
            list_div = content_div.find(
                'div', {"class": "mw-search-result-heading"})
            if list_div is not None:
                link = list_div.find("a")['href']
                return _get_text((query[0], "https://en.wikipedia.org/%s" % link))
            else:
                return query[0], ""

        children = content_div.findChildren()
        text = []
        flag = False
        for child in children:
            if child.name == "h1":
                flag = True
                continue
            if child.name == 'h2':
                break
            if child.name != "div" and flag:
                text.append(child.text.strip().replace('\n', " "))
                for child in child.findChildren():
                    children.remove(child)
        return query[0], (" . ".join(text)).lower()
    return query[0], ""


def query_abstracts(file, outfile, emtpty_labels):
    with open(file, 'r') as f, open(outfile, 'w+') as fout, open(emtpty_labels, 'w+') as labels_still_not_found:
        urls = []
        for idx, line in enumerate(f):
            if idx >= 14440:
                id = int(line.strip().split('->')[0])
                line = '->'.join(line.strip().split('->')[1:])
                query = 'https://en.wikipedia.org/w/index.php?search=%s&ns0=1' % (
                    urllib.parse.quote(line.lower()))
                urls.append([id, query])
        results = ThreadPool(50).imap_unordered(_get_text, urls)
        output_dict = {}
        for id, content in results:
            output_dict[id] = content
        keys = list(output_dict.keys())
        sorted_index = np.argsort(keys)
        for index in sorted_index:
            content = output_dict[keys[index]]
            id = keys[index]
            if content == "":
                print("%s" % (id), file=labels_still_not_found)
            print("%s->%s" % (id, content), file=fout)
            print(id, end='\r')


if __name__ == '__main__':
    query_abstracts(sys.argv[1], sys.argv[2], sys.argv[3])
