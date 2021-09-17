#coding: utf8
import openpyxl
import sys
import random
import collections
import pickle
import os
##  python scripts/extract_persona_chat.py data/Persona-Chat/ペルソナ対話1-5000.xlsx data/exp202102/flat/persona_chat/raw/all

use_persona = False


def make_context(dialog, cur_n, max_len=130):
    lines = []
    ll = 0
    for n in range(cur_n, max(-1, cur_n - 3), -1):
        d = dialog[n]
        if d["utt"] is None:
            break
        if len(lines) % 2 == 0:
            spk = "[SPK2]"
        else:
            spk = "[SPK1]"
        ll += len(d["utt"])
        if ll > max_len:
            lines.append(spk + d["utt"][:(max_len - ll)])
            break
        lines.append(spk + d["utt"])

    return "[SEP]".join(lines[::-1])


def main():
    if use_persona:
        # 対話種別・補足情報付き
        template = "個性雑談:[SEP]{persona}[SEP]{cont}"
    else:
        # フラット
        template = "{cont}"
    srcdata = collections.defaultdict(list)
    dstdata = collections.defaultdict(list)
    fname = sys.argv[1]
    outfname = sys.argv[2]
    os.makedirs(outfname, exist_ok=True)

    wb = openpyxl.load_workbook(fname, data_only=True)
    ws = wb["ペルソナリスト"]

    persona_data = {}
    for lines in ws:
        if lines[0].value == "No":
            continue

        persona = {}
        pid, persona["A"], persona["B"] = [v.value for v in lines][1:4]
        persona_data[pid] = persona

    ws = wb["対話"]
    dialog_data = collections.defaultdict(list)
    dialog = []
    prev_sid = "PP1"
    for lines in ws:
        lines = list(lines)
        if lines[0].value == "No":
            continue
        sid, spk, utt = [v.value for v in lines][1:4]
        utt = utt.replace("\n", "")
        if prev_sid != sid:
            dialog_data[prev_sid] = dialog[:]
            persona_data[prev_sid]["dialog"] = dialog[:]
            dialog = []
            prev_sid = sid
        dialog.append({"pid": sid, "spk": spk, "utt": utt})
    dialog_data[prev_sid] = dialog[:]
    persona_data[prev_sid]["dialog"] = dialog[:]

    #print(sit_data.keys())
    total_lines = 0
    dtype = "train"
    for sid, dialog in dialog_data.items():
        #feel = sit_data[sid]["feel"]
        #sit_sent = sit_data[sid]["sit_sent"]
        #print(dialog)
        #print(len(dialog))
        _src_data = []
        _dst_data = []
        for n in range(len(dialog)):
            if n == 0:
                continue
            if use_persona:
                cont = make_context(dialog, n - 1, 90)
            else:
                cont = make_context(dialog, n - 1)
            dst = dialog[n]["utt"]
            if use_persona:
                spk = dialog[n]["spk"]
                src = template.format(persona=persona_data[sid][spk].replace("\n", ""), cont=cont)
            else:
                src = template.format(cont=cont)

            _src_data.append(src)
            _dst_data.append(dst)
        ## persona_chat15k
        if total_lines > 15000:
            dtype = "rest"
        if total_lines > 50000:
            dtype = "valid"
        if total_lines > 53000:
            dtype = "test"
        total_lines += len(_src_data)
        dstdata[dtype] += _dst_data
        srcdata[dtype] += _src_data
    #print(len(srcdata))

    for dtype in srcdata.keys():
        idx = list(range(len(srcdata[dtype])))
        random.seed(0)
        random.shuffle(idx)
        _srcdata = [srcdata[dtype][i] for i in idx]
        _dstdata = [dstdata[dtype][i] for i in idx]

        outfname = sys.argv[2]
        with open(outfname + "/" + dtype + ".src", "w") as f:
            f.write("\n".join(_srcdata))

        with open(outfname + "/" + dtype + ".dst", "w") as f:
            f.write("\n".join(_dstdata))

    with open(outfname + "/persona_data.pkl", "wb") as f:
        pickle.dump(persona_data, f)


if __name__ == "__main__":
    main()
