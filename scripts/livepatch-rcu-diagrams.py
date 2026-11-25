#!/usr/bin/env python3
"""Diagrams for the "When livepatch needs help from RCU" post.

states     the per-task patch_state through one transition
scan       the four functions that drive a transition and what each walks
switch     klp_try_switch_task as a decision flow
points     every place a task can be switched
exitwin    the exit path of a self-reaping task and what can see it
tasksrcu   the five phases of a Tasks RCU grace period and which phase covers which stretch
trace      Tasks Trace RCU: per-CPU counters, flip and drain
fix        the proposed livepatch change laid over the same exit path
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "livepatch-rcu")
os.makedirs(OUT, exist_ok=True)

BLUE, GREEN, ORANGE, RED, PURPLE, GREY, DARK = "#2c5f8a", "#2e7d52", "#b8681c", "#c0392b", "#6b4c9a", "#888", "#333"
FONT = 'font-family="Helvetica, Arial, sans-serif"'
MONO = 'font-family="Menlo, Consolas, monospace"'


def head(w, h, label):
    marks = "".join(
        f'<marker id="{i}" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{c}"/></marker>'
        for i, c in (("ar", BLUE), ("arG", GREY), ("arR", RED), ("arGr", GREEN), ("arO", ORANGE), ("arP", PURPLE)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT} '
            f'role="img" aria-label="{label}">\n<defs>{marks}</defs>\n<rect width="{w}" height="{h}" fill="#fff"/>\n')


def txt(x, y, t, size=12, fill="#444", anchor="middle", weight="normal", mono=False):
    f = MONO if mono else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" {f}>{t}</text>\n')


def box(x, y, w, h, fill="#fff", stroke=GREY, rx=5, sw=1.5, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" fill-opacity="{op}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def line(x1, y1, x2, y2, stroke=GREY, sw=1.5, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>\n'


def title(w, t):
    return txt(w / 2, 26, t, 16, DARK, weight="bold")


def save(name, svg):
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write(svg + "</svg>\n")


# ---------------------------------------------------------------------------
def states():
    W, H = 900, 400
    s = head(W, H, "One task's patch_state goes from idle to the initial state, to the target state, and back to idle; "
                   "the global switch is func transition")
    s += title(W, "One task's patch_state during a transition that enables a patch")
    xs = [30, 250, 470, 690]
    names = [("KLP_TRANSITION_IDLE", "-1", GREY), ("KLP_TRANSITION_UNPATCHED", "0", ORANGE),
             ("KLP_TRANSITION_PATCHED", "1", GREEN), ("KLP_TRANSITION_IDLE", "-1", GREY)]
    for x, (n, v, c) in zip(xs, names):
        s += box(x, 70, 180, 64, "#fff", c, sw=2)
        s += txt(x + 90, 95, n, 11, c, weight="bold", mono=True)
        s += txt(x + 90, 118, f"patch_state = {v}", 12, "#444")
    labs = ["klp_init_transition()", "task switches", "klp_complete_transition()"]
    subs = ["initial state: old code", "to the target state: new code", "back to idle"]
    for i in range(3):
        s += line(xs[i] + 180, 102, xs[i + 1], 102, BLUE, 2, marker="ar")
        s += txt(xs[i] + 215, 62, labs[i], 11, BLUE, mono=True) if False else ""
        s += txt((xs[i] + 180 + xs[i + 1]) / 2, 150, labs[i], 10.5, BLUE, mono=True)
        s += txt((xs[i] + 180 + xs[i + 1]) / 2, 168, subs[i], 10.5, "#555")
    s += box(30, 200, 840, 64, "#f4f7fb", BLUE, 6, 1.4)
    s += txt(450, 224, "While func->transition is true, every call of a patched function goes through klp_ftrace_handler()", 12.5, DARK)
    s += txt(450, 247, "which reads current->patch_state: UNPATCHED runs the old code, anything else runs the new code", 12.5, DARK)
    s += box(30, 290, 410, 80, "#fff", ORANGE, 6, 1.5)
    s += txt(235, 314, "TIF_PATCH_PENDING", 12.5, ORANGE, weight="bold", mono=True)
    s += txt(235, 336, "set by klp_start_transition() on every task", 12, "#444")
    s += txt(235, 356, "means: this task still has to move to the target state", 12, "#444")
    s += box(460, 290, 410, 80, "#fff", GREEN, 6, 1.5)
    s += txt(665, 314, "klp_transition_patch, klp_target_state", 12.5, GREEN, weight="bold", mono=True)
    s += txt(665, 336, "one global patch and one global target", 12, "#444")
    s += txt(665, 356, "so one transition runs at a time", 12, "#444")
    save("states", s)


# ---------------------------------------------------------------------------
def scan():
    W, H = 900, 470
    s = head(W, H, "The four functions of a transition each walk the task list under tasklist_lock, and the completion "
                   "function retries about once a second")
    s += title(W, "The transition scan: four task-list walks")
    rows = [
        ("klp_init_transition()", "sets patch_state = initial state on every task", "then func->transition = true"),
        ("klp_start_transition()", "sets TIF_PATCH_PENDING on every task not yet in the target state", "enables the scheduler hook"),
        ("klp_try_complete_transition()", "calls klp_try_switch_task() on every task", "tasks left over: run again in about 1 s from a delayed work item"),
        ("klp_complete_transition()", "resets every task to IDLE", "func->transition = false"),
    ]
    y = 56
    for i, (fn, a, b) in enumerate(rows):
        s += box(30, y, 640, 74, "#fff", BLUE, 6, 1.6)
        s += txt(46, y + 24, fn, 13, BLUE, "start", "bold", True)
        s += txt(46, y + 46, a, 12, "#444", "start")
        s += txt(46, y + 64, b, 12, "#777", "start")
        s += box(690, y + 14, 180, 46, "#fff3e6", ORANGE, 5, 1.5)
        s += txt(780, y + 34, "for_each_process_thread", 10.5, ORANGE, weight="bold", mono=True)
        s += txt(780, y + 50, "under tasklist_lock", 11, ORANGE)
        if i < 3:
            s += line(350, y + 74, 350, y + 96, BLUE, 2, marker="ar")
        y += 96
    s += line(30, 262, 14, 262, RED, 2) if False else ""
    s += f'<path d="M 670 238 C 730 238 730 214 700 214" fill="none" stroke="{RED}" stroke-width="0"/>\n'
    s += txt(450, 446, "Idle tasks are not on the task list, so every walk also visits idle_task(cpu) for each CPU", 12, "#555")
    save("scan", s)


# ---------------------------------------------------------------------------
def switch():
    W, H = 900, 560
    s = head(W, H, "klp_try_switch_task returns success if the task already is in the target state, otherwise it checks "
                   "the task is off CPU and its reliable stack holds no patched function, then switches it")
    s += title(W, "klp_try_switch_task(): when a task may move to the target state")
    steps = [
        ("patch_state == klp_target_state ?", "yes: done, return true"),
        ("reliable stack traces available ?", "no: return false, never switches this way"),
        ("task running on a CPU (not current) ?", "yes: -EBUSY, try again later"),
        ("stack_trace_save_tsk_reliable() worked ?", "no: -EINVAL, try again later"),
        ("a patched function's code is on the stack ?", "yes: -EADDRINUSE, try again later"),
    ]
    y = 52
    for i, (q, no) in enumerate(steps):
        s += box(40, y, 360, 54, "#fff", BLUE, 6, 1.6)
        s += txt(220, y + 32, q, 12.5, DARK, weight="bold", mono=False)
        s += box(480, y + 4, 390, 46, "#fff4f2" if i else "#eef8f1", RED if i else GREEN, 5, 1.3)
        s += txt(675, y + 32, no, 12, RED if i else GREEN)
        s += line(400, y + 27, 480, y + 27, RED if i else GREEN, 1.8, marker="arR" if i else "arGr")
        s += txt(440, y + 20, "yes" if i in (0, 2, 4) else "no", 10.5, "#555")
        if i < 4:
            s += line(220, y + 54, 220, y + 78, BLUE, 2, marker="ar")
            s += txt(238, y + 70, "no" if i in (0, 2, 4) else "yes", 10.5, "#555", "start")
        y += 78
    s += line(220, y - 24, 220, y + 6, BLUE, 2, marker="ar")
    s += txt(238, y - 6, "no", 10.5, "#555", "start")
    s += box(40, y + 6, 360, 62, "#eef8f1", GREEN, 6, 2)
    s += txt(220, y + 30, "clear_tsk_thread_flag(TIF_PATCH_PENDING)", 12, GREEN, weight="bold", mono=True)
    s += txt(220, y + 52, "task->patch_state = klp_target_state", 12, GREEN, weight="bold", mono=True)
    s += txt(675, y + 40, "A task found running is not switched by this code at all:", 12, "#555")
    s += txt(675, y + 58, "it waits for one of the other switch points", 12, "#555")
    save("switch", s)


# ---------------------------------------------------------------------------
def points():
    W, H = 900, 440
    s = head(W, H, "A task can be switched at kernel exit, in the idle loop, on a freezable sleep in the scheduler, "
                   "by the scanner, or by inheriting at fork")
    s += title(W, "Where a task's patch_state can change, and what nudges it")
    s += box(350, 170, 200, 70, "#f4f7fb", BLUE, 8, 2)
    s += txt(450, 200, "a task with", 13, DARK, weight="bold")
    s += txt(450, 220, "TIF_PATCH_PENDING", 13, BLUE, weight="bold", mono=True)
    items = [
        (30, 56, "return to user space", "klp_update_patch_state()", "no stack check: no kernel function live", GREEN),
        (30, 160, "idle loop", "klp_update_patch_state()", "no stack check", GREEN),
        (30, 264, "fork", "klp_copy_process()", "child copies parent's patch_state", GREEN),
        (590, 56, "sleep with TASK_FREEZABLE", "__klp_sched_try_switch()", "from __schedule(), stack check", ORANGE),
        (590, 160, "periodic scan, about 1 s", "klp_try_complete_transition()", "task must be off CPU, stack check", ORANGE),
        (590, 264, "nudge only", "klp_send_signals()", "every 15 rounds: pushes a task toward a switch point", PURPLE),
    ]
    for x, y, a, b, c, col in items:
        s += box(x, y, 280, 80, "#fff", col, 6, 1.8)
        s += txt(x + 140, y + 24, a, 13, col, weight="bold")
        s += txt(x + 140, y + 46, b, 11, DARK, mono=True)
        s += txt(x + 140, y + 66, c, 11.5, "#555")
        sx = x + 280 if x < 450 else x
        s += line(sx, y + 40, 350 if x < 450 else 550, 205, col, 1.4, dash="4 3")
    s += txt(450, 380, "green: no stack check is needed. orange: a reliable stack trace must show no patched function. purple: changes nothing itself.", 12, "#555")
    s += txt(450, 402, "A kernel thread that never sleeps and never returns to user space can only be switched by the orange rows.", 12, "#555")
    save("points", s)


# ---------------------------------------------------------------------------
EV = [("exit_tasks_rcu_start()", 60), ("exit_notify() unhashes", 250), ("rest of do_exit()", 450),
      ("preempt_disable()", 590), ("exit_tasks_rcu_finish()", 740), ("do_task_dead()", 860)]


def timeline(s, y):
    s += line(40, y, 880, y, DARK, 2.5)
    for i, (n, x) in enumerate(EV):
        s += line(x, y - 8, x, y + 8, DARK, 2)
        anc = "start" if i == 0 else "end" if i == len(EV) - 1 else "middle"
        s += txt(x if anc == "middle" else (40 if i == 0 else 880), y - 16 - (14 if i % 2 else 0), n, 10.5, DARK, anc, mono=True)
    return s


def bar(s, y, x0, x1, label, col, fill, dash=None):
    s += box(x0, y, x1 - x0, 30, fill, col, 4, 1.6, dash=dash)
    s += txt((x0 + x1) / 2, y + 20, label, 11.5, col, weight="bold")
    return s


def exitwin():
    W, H = 900, 330
    s = head(W, H, "A self-reaping task leaves the task list in exit_notify but keeps running until its final schedule")
    s += title(W, "The exit path of a self-reaping task, and who can see it")
    s = timeline(s, 90)
    s = bar(s, 120, 60, 250, "on the task list", BLUE, "#eaf1f8")
    s = bar(s, 160, 60, 740, "on a Tasks RCU exit list", GREEN, "#eaf6ee")
    s = bar(s, 200, 590, 860, "non-preemptible", PURPLE, "#f1ecf8")
    s = bar(s, 240, 250, 860, "invisible to for_each_process_thread, still running kernel code", RED, "#fdeceb", "6 3")
    s += txt(450, 306, "A walk of the task list finds only the task-list bar. The bottom bar is the stretch it cannot see.", 12, "#555")
    save("exitwin", s)


def fix():
    W, H = 900, 400
    s = head(W, H, "The proposed livepatch change visits the exit lists at every step and waits for all CPUs to switch, "
                   "which together cover the whole exit path")
    s += title(W, "The proposed livepatch change laid over the exit path")
    s = timeline(s, 90)
    s = bar(s, 120, 60, 250, "task-list walk (today)", BLUE, "#eaf1f8")
    s = bar(s, 160, 60, 740, "walk of the exit lists (new, patch 1 and 2)", GREEN, "#eaf6ee")
    s = bar(s, 200, 590, 860, "wait for tasks already in here (new)", PURPLE, "#f1ecf8")
    s += txt(450, 262, "the walks find tasks up to exit_tasks_rcu_finish(); the wait, run after them, covers tasks already past it", 12.5, GREEN, weight="bold")
    s += box(40, 286, 820, 90, "#f4f7fb", BLUE, 6, 1.4)
    s += txt(450, 310, "klp_synchronize_transition() runs in klp_init_transition() before func->transition is set,", 12, DARK)
    s += txt(450, 330, "and in klp_try_complete_transition() before klp_complete_transition().", 12, DARK)
    s += txt(450, 352, "The second one is what keeps the old patch module from being freed under a task still inside it.", 12, DARK)
    save("fix", s)


# ---------------------------------------------------------------------------
def tasksrcu():
    W, H = 900, 400
    s = head(W, H, "The five phases of a Tasks RCU grace period; the exit lists are consulted in postscan and the "
                   "closing synchronize_rcu waits out the non-preemptible tail")
    s += title(W, "A Tasks RCU grace period and the exit path")
    ph = [("1 pregp", "synchronize_rcu()"), ("2 pertask", "every task on the list"), ("3 postscan", "every task on an exit list"),
          ("4 holdout loop", "poll until none left"), ("5 postgp", "synchronize_rcu()")]
    for i, (a, b) in enumerate(ph):
        x = 24 + i * 174
        col = GREEN if i in (2,) else PURPLE if i == 4 else BLUE
        s += box(x, 56, 160, 64, "#fff", col, 6, 1.8)
        s += txt(x + 80, 82, a, 13, col, weight="bold")
        s += txt(x + 80, 104, b, 11, "#555")
        if i < 4:
            s += line(x + 160, 88, x + 174, 88, BLUE, 2, marker="ar")
    s = timeline(s, 200)
    s = bar(s, 228, 60, 250, "phase 2 finds these", BLUE, "#eaf1f8")
    s = bar(s, 266, 60, 740, "phase 3 finds these", GREEN, "#eaf6ee")
    s = bar(s, 304, 590, 860, "phase 5 waits these out", PURPLE, "#f1ecf8")
    s += txt(450, 366, "Between them the phases cover the exit path from the first exit list insertion to the last switch.", 12, "#555")
    save("tasksrcu", s)


# ---------------------------------------------------------------------------
def trace():
    W, H = 900, 480
    s = head(W, H, "Tasks Trace RCU readers bump per-CPU counters; a grace period flips which set new readers use "
                   "and waits until the old set's lock and unlock sums match")
    s += title(W, "Tasks Trace RCU: per-CPU counters, flip, drain")
    panels = [
        ("1  readers running", [(2, 1), (1, 1)], [(0, 0), (0, 0)], "A", "lock sum 3, unlock sum 2: one reader inside"),
        ("2  grace period starts", [(2, 1), (1, 1)], [(1, 0), (0, 0)], "B", "new readers use set B; wait for set A"),
        ("3  last old reader leaves", [(2, 1), (1, 2)], [(1, 0), (0, 0)], "B", "set A: lock sum 3 = unlock sum 3: done"),
    ]
    for i, (t, a, b, cur, note) in enumerate(panels):
        x = 20 + i * 292
        s += box(x, 50, 276, 380, "#fff", GREY, 6, 1.4)
        s += txt(x + 138, 74, t, 13, DARK, weight="bold")
        for j in range(2):
            y = 104 + j * 150
            s += txt(x + 16, y + 6, f"CPU {j}", 12, "#444", "start", "bold")
            for k, (nm, vals, col) in enumerate((("set A", a[j], BLUE), ("set B", b[j], ORANGE))):
                yy = y + 14 + k * 54
                s += box(x + 16, yy, 244, 46, "#fff", col, 4, 1.5 if cur != nm[-1] or True else 1)
                s += txt(x + 28, yy + 19, nm, 11.5, col, "start", "bold")
                s += txt(x + 28, yy + 37, f"lock {vals[0]}    unlock {vals[1]}", 12, "#333", "start", mono=True)
        s += txt(x + 138, 408, note, 11, "#555")
    s += txt(450, 458, "A reader may unlock on a different CPU than it locked on; only the sums matter. (The real grace period scans the idle set first, then flips.)", 12, "#555")
    save("trace", s)


if __name__ == "__main__":
    for fn in (states, scan, switch, points, exitwin, tasksrcu, trace, fix):
        fn()
    print("wrote", sorted(os.listdir(OUT)))
