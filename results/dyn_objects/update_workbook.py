"""Collects the dynamic-object evaluation runs into results/dyn_objects/ and
rebuilds dyn_objects_results.xlsx (sheets Collisions, Exploration, Perception).

  ~/envs/xlsx/bin/python results/dyn_objects/update_workbook.py

Phantom (-matrixManipulation) values are taken from the existing workbook
(runs of 2026-11-21, jobs 28343207-28343210, before the perception metrics
existed). Rendered and both-flags runs are read from their train.log in
~/scratch/ans_runs/models/<exp>/ and copied here with their per-episode
files. The per-step trajectory logs (dump/<exp>/dyn_traj, ~35 MB per run)
stay on scratch.
Blue = values from train.log or the paper; black = formulas.
"""
import os
import re
import shutil
import sys

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "dyn_objects_results.xlsx")
RUNS = os.path.expanduser("~/scratch/ans_runs")
JOBS = {  # exp name -> Slurm job id (logs/full_runs_2026-11-26.txt)
    "dyn_render_known_n1": 28810032, "dyn_render_known_n3": 28810033,
    "dyn_render_random_n1": 28810034, "dyn_render_random_n3": 28814347,
    "dyn_both_known_n1": 28810056, "dyn_both_known_n3": 28814348,
    "dyn_both_random_n1": 28810060, "dyn_both_random_n3": 28814349,
}
ORDER = [("known", 1), ("random", 1), ("known", 3), ("random", 3)]
CONDS = [("phantom", "-matrixManipulation"), ("render", "-withRendering"),
         ("both", "-matrixManipulation -withRendering")]
LABEL = {"phantom": "phantom", "render": "rendered", "both": "both"}

BLUE = Font(color="0000FF")
HEAD = Font(bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="305496")
THIN = Side(style="thin")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
GREY = Font(color="595959")
CENTER = Alignment(horizontal="center")


# ----------------------------------------------------------------- inputs
def parse_log(path):
    txt = open(path).read()
    num = r"([-\d.]+|inf|nan)"

    def get(label):
        m = re.search(re.escape(label) + r"\s*" + num, txt)
        return float(m.group(1)) if m else None

    def last_of(header):
        block = txt.split(header, 1)[1].split("\n", 2)[1]
        return float([v for v in block.split(",") if v.strip()][-1])

    return {
        "events": get("contact events/eps:"),
        "contact": get("contact steps/eps:"),
        "blocked": get("blocked steps/eps:"),
        "yields": get("object yields/eps:"),
        "free": get("collision-free eps:") / 100.,
        "area": last_of("Final Exp Area:"),
        "ratio": last_of("Final Exp Ratio:"),
        "recall": get("object cell recall:"),
        "frame_det": get("frame detection rate:"),
        "in_view": get("steps in view/eps:"),
        "ghost_mean": get("ghost area/eps (mean over steps):"),
        "ghost_last": get("ghost area/eps (last step):"),
        "closest": get("closest approach/eps:"),
    }


def collect():
    """Copy each finished run's files here and parse its summary."""
    res = {}
    for exp, job in JOBS.items():
        log = os.path.join(RUNS, "models", exp, "train.log")
        if not os.path.exists(log) or "Dyn perception" not in open(log).read():
            print("skip (not finished):", exp)
            continue
        out = os.path.join(HERE, exp)
        os.makedirs(out, exist_ok=True)
        shutil.copy(log, out)
        for f in ["dyn_collisions.txt", "dyn_perception.txt",
                  "explored_area.txt", "explored_ratio.txt"]:
            shutil.copy(os.path.join(RUNS, "dump", exp, f), out)
        shutil.copy(os.path.join(RUNS, "logs", "{}-{}.out".format(exp, job)),
                    os.path.join(out, "slurm.out"))
        res[exp] = parse_log(log)
    return res


def phantom_from_workbook(wb):
    """Phantom rows of the original workbook (Collisions rows 5-8 = known1,
    random1, known3, random3; Exploration rows 7-10, same order). Works on
    the original layout and on a workbook already rebuilt by this script."""
    out = {}
    ws = wb["Collisions"]
    if ws["A4"].value == "Mode":  # original layout
        for r in range(5, 9):
            m, n = ws.cell(r, 1).value, ws.cell(r, 2).value
            out[(m, n)] = {k: ws.cell(r, c).value for k, c in
                           [("events", 3), ("contact", 4), ("blocked", 5),
                            ("yields", 6), ("free", 7)]}
        ex = wb["Exploration"]
        for r, key in zip(range(7, 11), ORDER):
            out[key]["area"] = ex.cell(r, 2).value
            out[key]["ratio"] = ex.cell(r, 7).value
        paper = (ex["B5"].value, ex["G5"].value)
        repro = (ex["B6"].value, ex["G6"].value)
    else:  # rebuilt layout: Condition, Mode, Objects, ...
        for r in range(5, 17):
            if ws.cell(r, 1).value != "phantom":
                continue
            key = (ws.cell(r, 2).value, ws.cell(r, 3).value)
            out[key] = {k: ws.cell(r, c).value for k, c in
                        [("events", 4), ("contact", 5), ("blocked", 6),
                         ("yields", 7), ("free", 8)]}
        ex = wb["Exploration"]
        for r in range(7, 19):
            if ex.cell(r, 1).value.startswith("phantom"):
                m, n = ex.cell(r, 1).value.split(", ")[1:3]
                key = (m, int(n.split()[0]))
                out[key]["area"] = ex.cell(r, 2).value
                out[key]["ratio"] = ex.cell(r, 7).value
        paper = (ex["B5"].value, ex["G5"].value)
        repro = (ex["B6"].value, ex["G6"].value)
    return out, paper, repro


# ----------------------------------------------------------------- output
def header(ws, row, names, widths):
    for c, (name, w) in enumerate(zip(names, widths), 1):
        cell = ws.cell(row, c, name)
        cell.font, cell.fill, cell.border = HEAD, FILL, BOX
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width = w
    ws.row_dimensions[row].height = 32


def put(ws, r, c, v, fmt="General", blue=False, center=True, bold=False):
    cell = ws.cell(r, c, v)
    cell.border = BOX
    cell.number_format = fmt
    if center:
        cell.alignment = CENTER
    if blue and v is not None and not (isinstance(v, str) and
                                       v.startswith("=")):
        cell.font = Font(color="0000FF", bold=bold)
    elif bold:
        cell.font = Font(bold=True)
    return cell


def notes(ws, row, lines):
    for i, t in enumerate(lines):
        ws.cell(row + i, 1, t).font = GREY


def title(ws, t, sub):
    ws["A1"] = t
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = sub
    ws["A2"].font = GREY


def rows_by_condition(phantom, new):
    for cond, flags in CONDS:
        for mode, n in ORDER:
            if cond == "phantom":
                yield cond, flags, mode, n, phantom[(mode, n)]
            else:
                exp = "dyn_{}_{}_n{}".format(cond, mode, n)
                if exp in new:
                    yield cond, flags, mode, n, new[exp]


def build(phantom, paper, repro, new):
    wb = openpyxl.Workbook()
    common = ("Gibson val, 994 episodes x 1000 steps, --dyn_block 1, "
              "speed 0.1 m/step, radius 0.25 m, same cached known routes")

    # Collisions
    ws = wb.active
    ws.title = "Collisions"
    title(ws, "Dynamic objects - collisions (mean per episode)", common)
    header(ws, 4, ["Condition", "Mode", "Objects", "Contact events",
                   "Contact steps", "Blocked steps", "Object yields",
                   "Collision-free eps", "Steps stuck per collision",
                   "Share of episode blocked"],
           [12, 9, 9, 14, 14, 14, 14, 16, 18, 18])
    ws["L4"], ws["M4"] = "Episode length (steps)", 1000
    ws["M4"].font = BLUE
    ws.column_dimensions["L"].width = 22
    r = 5
    for cond, _, mode, n, v in rows_by_condition(phantom, new):
        put(ws, r, 1, LABEL[cond])
        put(ws, r, 2, mode)
        put(ws, r, 3, n)
        for c, k in zip(range(4, 8), ["events", "contact", "blocked",
                                       "yields"]):
            put(ws, r, c, v[k], "0.00", blue=True)
        put(ws, r, 8, v["free"], "0.0%", blue=True)
        put(ws, r, 9, "=IFERROR(E{0}/D{0},0)".format(r), "0.00")
        put(ws, r, 10, "=F{}/$M$4".format(r), "0.0%")
        r += 1
    ws.freeze_panes = "A5"
    notes(ws, r + 2, [
        "Condition: phantom = -matrixManipulation (disks drawn into the "
        "planner's map, not rendered); rendered = -withRendering (boxes in "
        "RGB/depth); both = both flags",
        "Contact events: separate collisions (no contact -> contact)",
        "Contact steps: steps spent touching an object",
        "Blocked steps: steps whose move was undone (= contact steps with "
        "--dyn_block 1)",
        "Object yields: steps an object waited so it would not walk into the "
        "agent",
        "Collision-free eps: share of episodes with zero collisions",
        "Steps stuck per collision = Contact steps / Contact events; Share "
        "blocked = Blocked steps / episode length",
        "Blue = values from train.log (results/dyn_objects/*/train.log); "
        "black = formulas. Phantom: jobs 28343207-28343210 (2026-11-21); "
        "rendered / both: jobs 28810032-28814349 (2026-11-26/27)",
    ])

    # Exploration
    ex = wb.create_sheet("Exploration")
    title(ex, "Exploration after 1000 steps",
          "Gibson val, 994 episodes. Paper: Chaplot et al., Learning to "
          "Explore using Active Neural SLAM, ICLR 2020 (Gibson val, ANS)")
    header(ex, 4, ["Run", "Area (m²)", "Area vs paper", "Area vs paper (%)",
                   "Area vs reproduction", "Area vs reproduction (%)",
                   "Ratio", "Ratio vs paper", "Ratio vs reproduction",
                   "Area vs phantom (same mode, objects)",
                   "Ratio vs phantom (same mode, objects)"],
           [30, 11, 13, 15, 17, 19, 10, 13, 18, 20, 20])
    A, P, R = "0.000;-0.000;-", "0.0%;-0.0%;-", "0.0000;-0.0000;-"
    put(ex, 5, 1, "Paper (ANS)", center=False, bold=True)
    put(ex, 5, 2, paper[0], A, blue=True)
    put(ex, 5, 3, "=B5-$B$5", A)
    put(ex, 5, 4, "=B5/$B$5-1", P)
    put(ex, 5, 7, paper[1], R, blue=True)
    put(ex, 5, 8, "=G5-$G$5", R)
    put(ex, 6, 1, "Reproduction (no objects)", center=False, bold=True)
    put(ex, 6, 2, repro[0], A, blue=True)
    put(ex, 6, 7, repro[1], R, blue=True)
    phantom_row = {}
    r = 7
    rows = [(None, None, None, None, None)] + list(
        rows_by_condition(phantom, new))
    for i, (cond, _, mode, n, v) in enumerate(rows):
        if cond is None:  # reproduction row formulas
            rr = 6
        else:
            rr = r
            put(ex, rr, 1, "{}, {}, {} object{}".format(
                LABEL[cond], mode, n, "s" if n > 1 else ""), center=False)
            put(ex, rr, 2, v["area"], A, blue=True)
            put(ex, rr, 7, v["ratio"], R, blue=True)
            if cond == "phantom":
                phantom_row[(mode, n)] = rr
            else:
                pr = phantom_row[(mode, n)]
                put(ex, rr, 10, "=B{}-B{}".format(rr, pr), A)
                put(ex, rr, 11, "=G{}-G{}".format(rr, pr), R)
            r += 1
        put(ex, rr, 3, "=B{}-$B$5".format(rr), A)
        put(ex, rr, 4, "=B{}/$B$5-1".format(rr), P)
        put(ex, rr, 5, "=B{}-$B$6".format(rr), A)
        put(ex, rr, 6, "=B{}/$B$6-1".format(rr), P)
        put(ex, rr, 8, "=G{}-$G$5".format(rr), R)
        put(ex, rr, 9, "=G{}-$G$6".format(rr), R)
    ex.freeze_panes = "B5"
    notes(ex, r + 1, [
        "Blue = values from train.log (last of the 40 checkpoints = step "
        "1000) or from the paper; black = formulas.",
        "Negative = lower than the reference.",
    ])

    # Perception
    pe = wb.create_sheet("Perception")
    title(pe, "Dynamic objects - perception and map (mean per episode)",
          common + "; evaluation metrics added 2026-11-26")
    header(pe, 4, ["Condition", "Mode", "Objects", "Object cell recall",
                   "Frame detection rate", "Steps with object in view",
                   "Ghost area, mean over steps (m²)",
                   "Ghost area, last step (m²)", "Closest approach (m)"],
           [12, 9, 9, 15, 15, 16, 18, 16, 16])
    r = 5
    for cond, _, mode, n, v in rows_by_condition(phantom, new):
        if cond == "phantom":
            continue
        put(pe, r, 1, LABEL[cond])
        put(pe, r, 2, mode)
        put(pe, r, 3, n)
        put(pe, r, 4, v["recall"], "0.000", blue=True)
        put(pe, r, 5, v["frame_det"], "0.0%", blue=True)
        put(pe, r, 6, v["in_view"], "0.0", blue=True)
        put(pe, r, 7, v["ghost_mean"], "0.000", blue=True)
        put(pe, r, 8, v["ghost_last"], "0.000", blue=True)
        put(pe, r, 9, v["closest"], "0.000", blue=True)
        r += 1
    pe.freeze_panes = "A5"
    notes(pe, r + 1, [
        "Object cell recall: of the cells where a rendered object is seen in "
        "depth (obstacle with the boxes, free when re-rendered without them), "
        "the share the SLAM network's per-frame prediction marks as obstacle",
        "Frame detection rate: share of steps with an object in view where "
        ">= 50% of its cells were detected",
        "Ghost area: obstacle area in the agent's own map on the path objects "
        "have covered, where no object is now and that was not an obstacle "
        "before an object passed",
        "Closest approach: smallest agent-object centre distance per episode, "
        "where the agent's move took it (before a block undoes it); contact "
        "is < 0.35 m",
        "Phantom-only runs predate these metrics (not measured).",
        "Blue = values from train.log (results/dyn_objects/*/train.log).",
    ])
    return wb


if __name__ == "__main__":
    new = collect()
    phantom, paper, repro = phantom_from_workbook(
        openpyxl.load_workbook(XLSX))
    out = sys.argv[1] if len(sys.argv) > 1 else XLSX  # optional test path
    build(phantom, paper, repro, new).save(out)
    print("updated", out, "with", len(new), "new runs:", sorted(new))
