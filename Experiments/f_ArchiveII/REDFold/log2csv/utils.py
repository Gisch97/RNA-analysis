import pandas as pd
from glob import glob


def log2pd(file_path):
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                label = line[1:-1]
                next_line = f.readline().strip()
                next_next_line = f.readline().strip()
                rows.append(
                    {
                        "id": label,
                        "sequence": next_line,
                        "structure": next_next_line,
                    }
                )
    return pd.DataFrame(rows, columns=["id", "sequence", "structure"])


# TODO: copy the last F1 score function from sincfold metrics

MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]


def fold2bp(struc, xop="(", xcl=")"):
    openxs = []
    bps = []
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps


MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]


def fold2bp(struc, xop="(", xcl=")"):
    openxs = []
    bps = []
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps


def dot2bp(struc):
    bp = []
    for brackets in MATCHING_BRACKETS:
        bp = bp + fold2bp(struc, brackets[0], brackets[1])
    return list(sorted(bp))


def f1_score(ref_bp, pre_bp):
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1
    tp1 = 0
    for rbp in ref_bp:
        # add tolerance of +/- 1 position
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return f1


def dot2bp(struc):
    bp = []
    for brackets in MATCHING_BRACKETS:
        bp = bp + fold2bp(struc, brackets[0], brackets[1])
    return list(sorted(bp))


def f1_score(ref_bp, pre_bp):
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1
    tp1 = 0
    for rbp in ref_bp:
        # add tolerance of +/- 1 position
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return f1
