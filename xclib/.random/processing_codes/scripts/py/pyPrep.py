#! python3
import re
import sys
root = sys.argv[1]
file = root+"/"+sys.argv[1]

block = {
    "bullets": r"^\s*[*]\s+",
    "code": r"^[{][{][{]+\s*$",
    "empty": r"^\s*$",
    "heading": r"^\s*=+.*=+$",
    "indent": r"^[ \t]+",
}  # note that the priority is alphabetical

block_re = re.compile(r"|".join("(?P<%s>%s)" % kv
                                 for kv in sorted(block.items())))

with open(file,'r', encoding='latin1') as f:
    f_text = open(root+"/text.txt",'w')
    f_sa = open(root+"/sa.txt", 'w')
    f_title = open(root+"/title.txt", 'w')
    flag_page = False
    flag_title = False
    flag_id = False
    flag_text = False
    flag_seeAlso = False
    lines = 0
    text = ''
    sa = []
    idx = ''
    for line in f:
        line = line.strip()
        if not flag_page:
            if line == "<page>":
                flag_page = True
            continue
        elif line == "</page>":
            if sa != [] and text!="":
                f_text.write("%s->%s\n"%(idx,text))
                f_sa.write("%s->%s\n" % (idx, '-^-'.join(sa)))
                f_title.write("%s->%s\n" % (idx, title))
            flag_page = False
            flag_title = False
            flag_id = False
            flag_text = False
            flag_seeAlso = False
            lines = 0
            text = ''
            sa = []
            idx = ''

        else:
            if not flag_title:
                title = re.findall(r'\<title\>(.*)\</title\>', line)
                if  len(title) !=0:
                    title = title[0]
                    flag_title = True
                continue
            else:
                if not flag_id:
                    idx = re.findall(r'\<id\>(.*)\</id\>', line)
                    if  len(idx) >0:
                        idx = idx[0]
                        flag_id = True
                    continue
                else:
                    if not flag_text:
                        if line.find("<text xml:space=\"preserve\">") !=-1:
                            lines = 0
                            flag_text = True
                        continue
                    elif line == "<\\text>":
                        flag_text = False
                        continue
                    else:
                        if not flag_seeAlso:
                            if line == "==See also==":
                                flag_seeAlso = True
                                continue
                            else:
                                lines+=1
                                text += "%s" % line
                        elif line =="":
                            flag_seeAlso = False
                            flag_text = False
                        else:
                            if (line[0] == "*" and len(line)>2) and (line.find("{{") ==-1 and line.find("}}")==-1) :
                                line = line.replace("*","")
                                line = line.replace("[[","")
                                line = line.replace("[","")
                                line = line.replace("]]","")
                                line = line.replace("]", "")
                                sa.append(line.strip().lower())
