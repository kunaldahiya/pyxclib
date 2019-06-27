import xmltodict
import json
import _pickle as p
import sys
import re

db = re.compile(r"\[\[(.+?)\]\]")
dc = re.compile(r"\{\{(.+?)\}\}")

test_out = open("document_left_out.txt", "w")
def _clean_sa(sa):
    text = []
    txt = db.findall(sa.lower())+dc.findall(sa.lower())
    if len(txt) != 0:
        for _t in txt:
            _t = _t.split("|")
            if len(_t[0]) > 0:
                if _t[0][0] == ":":
                    _t[0] = _t[0][1:]
                if _t[0].find("portal") > -1:
                    _t = "%s:%s"%(_t[0].replace("-inline",""),_t[-1])
                elif _t[0].find("#") >-1:
                    _t = _t[0].split("#")[0]
                else:
                    _t = _t[0]
                text.extend([_t])
            else:
                continue
    return text


def _get_objects(obj):
    data = {}
    _id = obj['id']
    _title = obj['title']
    if _title is not None:
        _title = _title.lower()
        text = obj.get('revision', {}).get('text', {}).get(
            "#text", "").strip()
        if text !=  "":
            text = re.split(r"==.*?[Ss]ee [aA]lso.*?==", text)
            _text = text[0]
            if len(text) > 1:
                _SA_text = re.split(r"^==(?=[^=]).*?(References|Notes)", text[1].strip())[0]
                _SA = _clean_sa(_SA_text)
                if len(_SA) == 0:
                    print("%s,%s"%(_title,text[1]), file=test_out, flush=True)
            else:
                _text = text[0]    
                _SA = []
            data["id"] = _id
            data["text"] = _text.lower()
            data["SA"] = _SA 
            return _title, data
    return None, None
    
if __name__ == '__main__':
    with open(sys.argv[1],'r') as f:
        head = f.readline()
        paragraph = [head]
        flag = False
        data = {}
        idx = 0
        for line in f:
            if line == "  <page>\n":
                flag = True
            if flag:
                paragraph.append(line)
            if line == "  </page>\n":
                flag = False
                idx+=1
                if idx%1000 == 0:
                    paragraph.append("</mediawiki>")
                    o = xmltodict.parse(''.join(paragraph))['mediawiki']['page']
                    for obj in o:
                        _title, _data = _get_objects(obj)
                        if _title is not None:
                            data[_title] = _data
                    print(data[_title]["id"], _title, data[_title]["SA"])
                    paragraph = [head]
                print("Total datapoints are %d"%(idx), end='\r')
        p.dump(data, open(sys.argv[2], 'wb'))
