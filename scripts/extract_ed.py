#coding: utf8
import openpyxl
import sys
import random
import collections
import pickle
import os
use_sit = False


def make_context(dialog, cur_n, max_len=100):
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
    if use_sit:
        template = "共感雑談:[SEP]ID xx[SEP]{sit_sent}:{feel}[SEP]{cont}"
    else:
        template = "{cont}"
    srcdata = collections.defaultdict(list)
    dstdata = collections.defaultdict(list)
    fname = sys.argv[1]
    outfname = sys.argv[2]
    os.makedirs(outfname, exist_ok=True)
    wb = openpyxl.load_workbook(fname, data_only=True)
    ws = wb["状況文"]

    sit_data = {}
    for lines in ws:
        sid, feel, sit_sent = [v.value for v in lines][:3]
        sit_data[sid] = {"feel": feel, "sit_sent": sit_sent}
    #print(sit_data)
    ws = wb["対話"]
    dialog_data = collections.defaultdict(list)
    dialog = []
    prev_sid = "1_1"
    for lines in ws:
        lines = list(lines)
        if lines[0].value == "ID":
            continue
        sid, spk, utt = [v.value for v in lines][:3]
        if prev_sid != sid:
            dialog_data[prev_sid] = dialog[:]
            sit_data[prev_sid]["dialog"] = dialog[:]
            dialog = []
            prev_sid = sid
        dialog.append({"sid": sid, "spk": spk, "utt": utt})
    dialog_data[prev_sid] = dialog[:]
    sit_data[prev_sid]["dialog"] = dialog[:]

    #print(sit_data.keys())
    total_lines = 0
    dtype = "train"
    for sid, dialog in dialog_data.items():
        feel = sit_data[sid]["feel"]
        sit_sent = sit_data[sid]["sit_sent"]
        #print(dialog)
        #print(len(dialog))
        _src_data = []
        _dst_data = []
        for n in range(len(dialog)):
            if n == 0:
                continue
            cont = make_context(dialog, n - 1)
            dst = dialog[n]["utt"]
            if use_sit:
                src = template.format(sit_sent=sit_sent, feel=feel, cont=cont)
            else:
                src = template.format(cont=cont)

            _src_data.append(src)
            _dst_data.append(dst)
        #if total_lines > 15000:
        #    dtype = "rest"
        if total_lines > 50000:
            dtype = "valid"
        if total_lines > 55000:
            dtype = "test"
        total_lines += len(_src_data)
        dstdata[dtype] += _dst_data
        srcdata[dtype] += _src_data  #print(len(srcdata))

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

    with open(outfname + "/situation_data.pkl", "wb") as f:
        pickle.dump(sit_data, f)


if __name__ == "__main__":
    main()
