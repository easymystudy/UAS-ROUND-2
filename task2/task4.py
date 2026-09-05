# UAS-DTU Round 2 - Task 2 (Bonus Task)
# Walking a white circle to a black circle with a LIMITED RECEPTIVE FIELD
#
# The rule of this task :
# I am allowed to use CV on the full image ONLY ONCE, just to find where the
# white circle and the black circle are. After that I am NOT allowed to look
# at the whole image again. I can only see a small window around myself
# (width/4 x height/2) and I have to walk using only that.
#
# My steps (same as the pdf) :
# 1. find the mask of the white circle (and the black target) - one time only
# 2. crop the receptive field around the white circle
# 3. move 20 pixels per step, only up / down / left / right
# 4. cannot go through the red walls
# 5. save a snapshot after every step
# 6. join all the snapshots into a video
#

import cv2
import numpy as np
import os
from collections import deque  #datastructure adding or removing from both end (start or end)


ss = 20        # step size , the circle moves 20 pixels in one step
gr = 10        # grid , I plan on a small grid of 10 px to make it fast


# -------------------------------------------------------------------------
# PART 1 : the only time I am allowed to look at the full image
# -------------------------------------------------------------------------

def find_start_and_target(im):
    # the image has noise in it so I use blur first, otherwise the mask
    # comes out full of small dots
    bl = cv2.medianBlur(im, 5)                    # blur
    hsv = cv2.cvtColor(bl, cv2.COLOR_BGR2HSV)

    # white circle = i am here. white means saturation very low, value very high
    wh = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    # black circle = my target. everything dark
    bk = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))

    # clean the small dots
    kn = np.ones((5, 5), np.uint8)                # kernel
    wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN, kn)
    bk = cv2.morphologyEx(bk, cv2.MORPH_OPEN, kn)

    def biggest_centre(mk):
        cts, junk = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cts) == 0:
            return None, 0
        bg = max(cts, key=cv2.contourArea)        # biggest
        m = cv2.moments(bg)
        if m["m00"] == 0:
            return None, 0
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        rd = int(np.sqrt(cv2.contourArea(bg) / np.pi))        # radius
        return (x, y), rd

    st, mr = biggest_centre(wh)                   # start , my radius
    tg, junk = biggest_centre(bk)                 # target
    return st, tg, mr


# -------------------------------------------------------------------------
# PART 2 : this is the ONLY thing I can look at while walking
# -------------------------------------------------------------------------

def get_receptive_field(im, ce, rw, rh):
    """
    Cuts out a small window around me. This is the only part of the world
    I am allowed to see. It also tells me where this window sits, because
    I need that to change the small coordinates into big coordinates.
    """
    h, w = im.shape[:2]
    x1 = ce[0] - rw // 2
    y1 = ce[1] - rh // 2
    # if the window goes outside the image I push it back inside
    x1 = max(0, min(x1, w - rw))
    y1 = max(0, min(y1, h - rh))
    cp = im[y1:y1 + rh, x1:x1 + rw]               # crop
    return cp, x1, y1


def free_space_in_crop(cp):
    """
    In the crop, green = I can walk, red = wall.
    The image is noisy so I blur it first.
    I take everything that is NOT red as walkable (green, and also the
    white and black circles are walkable).
    """
    bl = cv2.medianBlur(cp, 5)
    hsv = cv2.cvtColor(bl, cv2.COLOR_BGR2HSV)
    rd = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))       # red
    rd2 = cv2.inRange(hsv, np.array([170, 90, 90]), np.array([180, 255, 255]))
    rd = cv2.bitwise_or(rd, rd2)
    # close the small holes made by the noise so the wall is one solid block
    rd = cv2.morphologyEx(rd, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fr = cv2.bitwise_not(rd)                      # free
    return fr


# -------------------------------------------------------------------------
# PART 3 : the memory map
# -------------------------------------------------------------------------
# I cannot see the whole world, but I am allowed to REMEMBER the places I
# have already seen. So I keep a map which starts completely unknown, and
# every step I paste whatever my receptive field showed me into it.
# When I plan, I treat the unknown places as if they are free (hoping there
# is a way there). If I walk there and find a wall, my memory gets corrected
# and next time I plan a different way. This is how a real robot does it.

UNKNOWN = 0
FREE = 1
WALL = 2


def remember(mm, fm, x1, y1, sf):                 # memory , free mask , safe
    """paste what I just saw into my memory map"""
    h, w = fm.shape
    # I make the walls a bit fatter by the radius of my circle, so that the
    # centre of my circle never goes so close that my body touches the wall
    fw = cv2.dilate((fm == 0).astype(np.uint8),
                    np.ones((sf * 2 + 1, sf * 2 + 1), np.uint8))      # fat wall
    pt = mm[y1:y1 + h, x1:x1 + w]                 # part
    pt[fw > 0] = WALL
    pt[(fw == 0)] = FREE


def plan_full_path(mm, me, tg):
    """
    BFS on my memory map from where I am to the target.
    The places I have not seen yet are counted as FREE (I am hopeful there is
    a way there). If I go there and find a wall, my memory gets corrected and
    I plan again. This is how a real robot explores.
    Gives back the full list of grid cells to follow.
    """
    h, w = mm.shape
    gh, gw = h // gr, w // gr

    def cell_is_ok(r, c):
        if r < 0 or c < 0 or r >= gh or c >= gw:
            return False
        pc = mm[r * gr:(r + 1) * gr, c * gr:(c + 1) * gr]         # piece
        return not (pc == WALL).any()

    sc = (me[1] // gr, me[0] // gr)               # start cell
    gc = (tg[1] // gr, tg[0] // gr)               # goal cell

    cf = {sc: None}                               # came from
    qu = deque([sc])                              # queue
    fd = False                                    # found
    while len(qu) > 0:
        cu = qu.popleft()                         # current
        if cu == gc:
            fd = True
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = (cu[0] + dr, cu[1] + dc)         # next
            if nx not in cf and cell_is_ok(nx[0], nx[1]):
                cf[nx] = cu
                qu.append(nx)

    if not fd:
        return None

    ph = []                                       # path
    cu = gc
    while cu is not None:
        ph.append(cu)
        cu = cf[cu]
    ph.reverse()
    return ph


def path_is_still_ok(mm, ph):
    """check that the next few cells of my plan are not walls now"""
    if ph is None or len(ph) < 2:
        return False
    gh, gw = mm.shape[0] // gr, mm.shape[1] // gr
    for ce in ph[1:6]:              # I only check a few cells ahead
        r, c = ce
        if r < 0 or c < 0 or r >= gh or c >= gw:
            return False
        pc = mm[r * gr:(r + 1) * gr, c * gr:(c + 1) * gr]
        if (pc == WALL).any():
            return False
    return True


def direction_to_cell(me, ce):
    """
    I can only move up / down / left / right, 20 px at a time.
    So to reach the next cell I first fix the x, then the y.
    (if I want to go diagonally I need two steps, like the pdf says)
    """
    wx = ce[1] * gr + gr // 2                     # want x
    wy = ce[0] * gr + gr // 2                     # want y
    if abs(wx - me[0]) >= ss // 2:
        if wx > me[0]:
            return (ss, 0)
        else:
            return (-ss, 0)
    if abs(wy - me[1]) >= ss // 2:
        if wy > me[1]:
            return (0, ss)
        else:
            return (0, -ss)
    return None                          # I am already on this cell


def can_i_stand_here(mm, pt, w, h):
    """check my memory : is this new place free ?"""
    x, y = pt
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mm[y, x] != WALL


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def run(ip, of, ms=400):                          # image path , out folder , max steps
    nm = os.path.splitext(os.path.basename(ip))[0]            # name
    im = cv2.imread(ip)
    ht, wd = im.shape[:2]                         # height , width

    # the receptive field is (width/4 x height/2) like the pdf says
    rw = wd // 4                                  # rf width
    rh = ht // 2                                  # rf height

    print("==================================================")
    print("IMAGE :", nm, " size =", wd, "x", ht)
    print("Receptive field size =", rw, "x", rh)

    # ---- the one and only global look ----
    me, tg, mr = find_start_and_target(im)
    if me is None or tg is None:
        print("could not find the white circle or the black circle")
        return
    print("White circle (start) =", me)
    print("Black circle (target)=", tg)
    print("my radius =", mr)
    print("--- from here I can only see my receptive field ---")

    sf = of + "/" + nm + "_steps"                 # snap folder
    if not os.path.exists(sf):
        os.makedirs(sf)

    # my memory of the world, everything unknown in the beginning
    mm = np.zeros((ht, wd), np.uint8)             # memory

    wk = [me]                                     # walked
    fm = []                                       # frames
    pl = None                                     # my plan
    pi = 1                                        # plan index
    sk = 0                                        # stuck count

    for sp in range(ms):                          # step
        # STEP A : look around me (this is all I am allowed to see)
        cp, x1, y1 = get_receptive_field(im, me, rw, rh)
        fr = free_space_in_crop(cp)               # free

        # STEP B : remember what I saw
        remember(mm, fr, x1, y1, mr + 2)

        # STEP C : save the snapshot of this moment
        fa = draw_snapshot(im, me, tg, wk, x1, y1, rw, rh, mr, sp, mm)      # frame
        cv2.imwrite(sf + "/step_%04d.png" % sp, fa)
        fm.append(fa)

        # have I reached ?
        if abs(me[0] - tg[0]) <= ss and abs(me[1] - tg[1]) <= ss:
            print("REACHED the black circle in", sp, "steps")
            break

        # STEP D : if I have no plan, or my plan is blocked now, plan again
        if not path_is_still_ok(mm, pl):
            pl = plan_full_path(mm, me, tg)
            pi = 1
            if pl is None:
                sk = sk + 1
                print("step", sp, ": no way found in my memory")
                if sk > 5:
                    print("I am stuck, stopping")
                    break
                continue
            sk = 0

        # skip the cells I am already standing on
        mv = None                                 # move
        while pi < len(pl):
            mv = direction_to_cell(me, pl[pi])
            if mv is not None:
                break
            pi = pi + 1
        if mv is None:
            pl = None
            continue

        np_ = (me[0] + mv[0], me[1] + mv[1])      # new place

        # STEP E : do not walk into a wall
        if not can_i_stand_here(mm, np_, wd, ht):
            mm[max(0, np_[1] - 3):np_[1] + 4,
               max(0, np_[0] - 3):np_[0] + 4] = WALL
            pl = None              # my plan was wrong, make a new one
            continue

        me = np_
        wk.append(me)

    print("total steps walked =", len(wk) - 1)
    print("snapshots saved in", sf)

    # ---- make the video ----
    vp = of + "/" + nm + "_video.mp4"             # video path
    if len(fm) > 0:
        fc = cv2.VideoWriter_fourcc(*"mp4v")      # fourcc
        vd = cv2.VideoWriter(vp, fc, 10.0, (wd, ht))          # video
        for f in fm:
            vd.write(f)
        # keep the last frame for a moment so we can see it reached
        for i in range(15):
            vd.write(fm[-1])
        vd.release()
        print("video saved in", vp)

    # also save the final picture with the full path drawn
    fi = im.copy()                                # final
    for i in range(1, len(wk)):
        cv2.line(fi, wk[i - 1], wk[i], (255, 0, 255), 3)
    cv2.circle(fi, wk[0], mr, (255, 255, 255), -1)
    cv2.circle(fi, wk[0], mr, (0, 0, 0), 2)
    cv2.circle(fi, tg, 12, (0, 255, 255), 3)
    cv2.imwrite(of + "/" + nm + "_final_path.png", fi)
    print("final path image saved")


def draw_snapshot(im, me, tg, wk, x1, y1, rw, rh, rd, sp, mm):
    """
    The snapshot shows what the robot can see. Everything outside the
    receptive field is made dark, because the robot cannot see it.
    """
    fa = (im * 0.25).astype(np.uint8)             # dark = I cannot see this
    fa[y1:y1 + rh, x1:x1 + rw] = im[y1:y1 + rh, x1:x1 + rw]

    # yellow box = my receptive field
    cv2.rectangle(fa, (x1, y1), (x1 + rw, y1 + rh), (0, 255, 255), 3)

    # the path I have walked till now
    for i in range(1, len(wk)):
        cv2.line(fa, wk[i - 1], wk[i], (255, 0, 255), 3)

    # me
    cv2.circle(fa, me, rd, (255, 255, 255), -1)
    cv2.circle(fa, me, rd, (0, 0, 0), 2)

    # writing on top
    cv2.putText(fa, "step " + str(sp), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(fa, "step " + str(sp), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(fa, "receptive field " + str(rw) + "x" + str(rh), (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(fa, "receptive field " + str(rw) + "x" + str(rh), (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return fa


def main():
    # I have only one maze image to work on, so I give the file name directly
    # instead of looping over a folder.
    input_image = "task2\images\maze1.jpg"
    output_folder = "task2\outputs"

    run(input_image, output_folder)



main()
