# UAS-DTU Round 2 - Task 1 
# Casualty Analysis using a Rover Guided by a UAV
# I am a beginner in opencv so I have written this in a simple way with the help of opensources.
# I have used only opencv, numpy and normal python maths.
# What I have to do :
# 1. make a mask of where the rover can go and where it cannot
# 2. find all the casualties , their position , age group , severity , elevation
# 3. make a path from orange triangle >> casualties >> purple triangle
# 4. find the score of that path
# 5. draw the path on the image and save it
# 6. find the total time
# 7. rank all the images by score and by time


import cv2
import numpy as np
import math
import os
import heapq  #it is queue algorithmm which i saw from google

# given details of the question which i have to use
ags = {"circle": 3, "star": 1, "square": 2}                                  # age score
agn = {"circle": "Children", "star": "Adults", "square": "Senior Citizens"}  # age name

#colour scores
svs = {"red": 3, "yellow": 2, "white": 1}                                    # severity score
svn = {"red": "Critical", "yellow": "Moderate", "white": "Safe"}             # severity name

# speed of rover on each green level
# level 0 = light green (grounded ) , level 1 = medium green (above the ground) , level 2 = darkest green( at top levl)
spd = [20.0, 15.0, 10.0]

#setting the pixel to which it will see we can adjust it, if i want the route in less time we will go with large value and most accurate with 1
bs = 1      # box size


# STEP 1 : finding the different colours

def get_hsv(im):
    # hsv is easier to use than bgr for finding colours
    return cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


def find_colour(im, l1, h1, l2=None, h2=None):
    # this makes a black and white mask , white where the colour is found
    hsv = get_hsv(im)
    mk = cv2.inRange(hsv, np.array(l1), np.array(h1))
    if l2 is not None:
        mk2 = cv2.inRange(hsv, np.array(l2), np.array(h2))
        mk = cv2.bitwise_or(mk, mk2)
    # remove the very small dots
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mk


def find_black(im):
    # black region = obstacle , rover cannot pass
    hsv = get_hsv(im)
    return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))


def find_water(im):
    # the big blue oval is a water body so the rover CANNOT go there.
    # I checked the colour of the water with print(img[y,x]) and got
    # hsv about (130, 230, 235) which is a very strong blue.
    hsv = get_hsv(im)
    wt = cv2.inRange(hsv, np.array([100, 180, 60]), np.array([140, 255, 255]))
    wt = cv2.morphologyEx(wt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return wt


def find_all_greens(im):
    # this gives me a list of green masks
    # first one is the lightest green (level 0) then darker and darker
    hsv = get_hsv(im)
    gr = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([90, 255, 255]))

    gp = im[gr > 0]                      # green pixels
    if len(gp) == 0:
        return [np.zeros(gr.shape, np.uint8)]

    # First I tried to separate the 3 greens using brightness ranges but when
    # I printed np.unique() I got hundreds of values because the edges of the
    # ellipses are blurred (anti aliasing). So now I find the main flat green
    # colours and then give every green pixel to the closest main colour.
    cl, hm = np.unique(gp.reshape(-1, 3), axis=0, return_counts=True)   # colours , how many
    be = hm > 0.01 * len(gp)             # big enough
    mc = cl[be].astype(np.int32)         # main colours
    mn = hm[be]                          # main counts

    # Now which green is level 0 , level 1 , level 2 ?
    # The elevations are drawn like circles inside circles , so the ground
    # covers the biggest area and the higher levels cover smaller areas.
    # So I sort by area , biggest area = level 0.
    mc = mc[np.argsort(-mn)]

    # for every green pixel find which main colour is nearest
    df = gp.astype(np.int32)[:, None, :] - mc[None, :, :]      # difference
    nr = np.argmin((df ** 2).sum(axis=2), axis=1)              # nearest

    ay, ax = np.nonzero(gr)              # all y , all x
    gm = []                              # green masks
    for i in range(len(mc)):
        om = np.zeros(gr.shape, np.uint8)          # one mask
        pk = (nr == i)                             # picked
        om[ay[pk], ax[pk]] = 255
        gm.append(om)
    return gm

# STEP 2 : finding what shape it is

def what_shape(ct):
    # circle = child , star = adult , square = senior , triangle = start/end marker
    ln = cv2.arcLength(ct, True)                              # length
    cr = len(cv2.approxPolyDP(ct, 0.02 * ln, True))           # corners
    ar = cv2.contourArea(ct)                                  # area
    if ar < 1:
        return "unknown"

    # a star is very spiky so a lot of its outer box is empty.
    # solidity = area / area of the convex hull
    ha = cv2.contourArea(cv2.convexHull(ct))                  # hull area
    if ha > 0:
        sl = ar / ha                                          # solidity
    else:
        sl = 1.0

    if sl < 0.75 or cr >= 9:
        return "star"
    if cr == 3:
        return "triangle"
    if cr == 4:
        return "square"

    # roundness = 4 * pi * area / (perimeter * perimeter) , it is 1 for a circle
    if ln > 0:
        rd = 4 * math.pi * ar / (ln * ln)                     # roundness
    else:
        rd = 0
    if rd > 0.7:
        return "circle"
    return "square"


def find_shapes_in_mask(mk, sa):
    # returns a list like [(shape name, (x,y)), ...]
    cts, junk = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ans = []
    for c in cts:
        if cv2.contourArea(c) < sa:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        ans.append((what_shape(c), (cx, cy)))
    return ans


def find_elevation(pt, gm, sz=14):
    # the casualty shape is drawn ON the green so I cannot see the green under it.
    # So I look at a small square around the shape and see which green is most there.
    x, y = pt
    ht, wd = gm[0].shape
    x1 = max(0, x - sz)
    y1 = max(0, y - sz)
    x2 = min(wd, x + sz)
    y2 = min(ht, y + sz)

    cn = []                                      # counts
    for m in gm:
        cn.append(int(np.count_nonzero(m[y1:y2, x1:x2])))

    if sum(cn) == 0:
        if sz < 100:
            return find_elevation(pt, gm, sz * 3)   # look in a bigger square
        return 0
    return int(np.argmax(cn))


# STEP 3 : path finding with A star

def get_speed(lv):
    if lv < len(spd):
        return spd[lv]
    return spd[-1]      # if there are more than 3 greens use slowest


class MapForRover:
    # I make a smaller grid of the image so that A star is fast

    def __init__(self, cg, lm):
        self.fr = cg[::bs, ::bs] > 0              # free
        self.lv = lm[::bs, ::bs]                  # level
        self.rw, self.cw = self.fr.shape          # rows , cols

    def point_to_cell(self, pt):
        return (pt[1] // bs, pt[0] // bs)         # (row, col)

    def cell_to_point(self, cel):
        return (cel[1] * bs, cel[0] * bs)         # (x, y)

    def is_free(self, cel):
        r, c = cel
        if r < 0 or c < 0 or r >= self.rw or c >= self.cw:
            return False
        return self.fr[r, c]

    def closest_free_cell(self, cel):
        # if the point is on an obstacle or on a shape , move it to the nearest free cell
        if self.is_free(cel):
            return cel
        for rg in range(1, 80):                   # ring
            for d in range(-rg, rg + 1):
                for sd in (-rg, rg):              # side
                    t1 = (cel[0] + d, cel[1] + sd)
                    t2 = (cel[0] + sd, cel[1] + d)
                    if self.is_free(t1):
                        return t1
                    if self.is_free(t2):
                        return t2
        return cel

    def find_path(self, sc, ec):                  # start cell , end cell
        # normal A star , returns (list of cells, distance in px, time in sec)
        sc = self.closest_free_cell(sc)
        ec = self.closest_free_cell(ec)
        if sc == ec:
            return [sc], 0.0, 0.0

        mv = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # moves
        tc = [(0.0, sc)]                          # to check
        cf = {}                                   # came from
        cs = {sc: 0.0}                            # cost so far
        ad = set()                                # already done

        while len(tc) > 0:
            junk, nw = heapq.heappop(tc)          # now
            if nw in ad:
                continue
            ad.add(nw)
            if nw == ec:
                break

            for dr, dc in mv:
                nx = (nw[0] + dr, nw[1] + dc)     # next
                if not self.is_free(nx):
                    continue
                sd = math.hypot(dr, dc) * bs      # step distance
                nc = cs[nw] + sd                  # new cost
                if nc < cs.get(nx, 999999999):
                    cs[nx] = nc
                    cf[nx] = nw
                    gs = math.hypot(nx[0] - ec[0], nx[1] - ec[1]) * bs    # guess
                    heapq.heappush(tc, (nc + gs, nx))

        if ec not in cf:
            return None, float("inf"), float("inf")    # no path found

        # go backwards to make the path
        ph = [ec]                                 # path
        while ph[-1] != sc:
            ph.append(cf[ph[-1]])
        ph.reverse()

        td = 0.0                                  # total distance
        tt = 0.0                                  # total time
        for i in range(1, len(ph)):
            a = ph[i - 1]
            b = ph[i]
            d = math.hypot(a[0] - b[0], a[1] - b[1]) * bs
            td = td + d
            tt = tt + d / get_speed(int(self.lv[b[0], b[1]]))
        return ph, td, tt


# doing everything for one image

def do_one_image(ip, of):                         # image path , output folder
    nm = os.path.splitext(os.path.basename(ip))[0]            # name
    im = cv2.imread(ip)
    if im is None:
        print("cannot open", ip)
        return None

    ht, wd = im.shape[:2]                         # height , width
    print("")
    print("==================================================================")
    print("IMAGE :", nm, "  size =", wd, "x", ht)
    print("==================================================================")

    # ---- masks ----
    gm = find_all_greens(im)                      # green masks
    bm = find_black(im)                           # black mask
    wm = find_water(im)                           # water mask
    print("number of green shades found =", len(gm))

    ag = np.zeros((ht, wd), np.uint8)             # all green
    for m in gm:
        ag = cv2.bitwise_or(ag, m)

    # the shapes are drawn on top of the green so those pixels are not green,
    # but the rover can still stand there , so I add them back
    rm = find_colour(im, [0, 120, 90], [8, 255, 255], [170, 120, 90], [180, 255, 255])   # red
    ym = find_colour(im, [20, 90, 150], [34, 255, 255])                                  # yellow
    wtm = find_colour(im, [0, 0, 200], [180, 40, 255])                                   # white
    om = find_colour(im, [9, 120, 120], [19, 255, 255])                                  # orange
    pm = find_colour(im, [125, 60, 120], [160, 179, 255])                                # purple

    sm = np.zeros((ht, wd), np.uint8)             # shapes mask
    for m in [rm, ym, wtm, om, pm]:
        sm = cv2.bitwise_or(sm, m)

    cg = cv2.bitwise_or(ag, sm)                   # can go
    cg[bm > 0] = 0      # black obstacle
    cg[wm > 0] = 0      # water body , rover cannot swim
    cg = cv2.morphologyEx(cg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cg[bm > 0] = 0      # doing it again so the closing does not eat the wall
    cg[wm > 0] = 0

    ng = cv2.bitwise_not(cg)                      # cannot go

    # OUTPUT 1 : the mask image
    mp = np.zeros((ht, wd, 3), np.uint8)          # mask picture
    mp[cg > 0] = (255, 255, 255)
    cv2.imwrite(of + "/" + nm + "_1_mask.png", mp)

    # a colour version so it is easy to see what is what
    cp = np.zeros((ht, wd, 3), np.uint8)          # colour picture
    ngr = [(120, 255, 120), (60, 180, 60), (20, 100, 20)]     # nice greens
    for i in range(len(gm)):
        cp[gm[i] > 0] = ngr[min(i, 2)]
    cp[bm > 0] = (0, 0, 255)        # red = wall
    cp[wm > 0] = (255, 100, 0)      # blue = water
    cv2.imwrite(of + "/" + nm + "_1b_terrain.png", cp)

    # this map tells which green level every pixel is on , needed for the speed
    lm = np.zeros((ht, wd), np.uint8)             # level map
    for i in range(len(gm)):
        lm[gm[i] > 0] = i

    # ---- start and end ----
    # NOTE : there is also an orange CIRCLE casualty in the image, so I must take
    # only the TRIANGLE otherwise the start point comes out wrong.
    sp = None                                     # start point
    ep = None                                     # end point
    for sh, ce in find_shapes_in_mask(om, 200):
        if sh == "triangle":
            sp = ce
    for sh, ce in find_shapes_in_mask(pm, 200):
        if sh == "triangle":
            ep = ce

    if sp is None or ep is None:
        print("could not find the start or the end triangle")
        return None

    print("Start (orange triangle)      =", sp)
    print("Destination (purple triangle)=", ep)

    # ---- casualties ----
    cl = []                                       # casualty list
    for cn, cm in [("red", rm), ("yellow", ym), ("white", wtm)]:      # colour name , colour mask
        for sh, ce in find_shapes_in_mask(cm, 120):
            if sh not in ags:
                continue         # triangle or unknown , it is not a casualty
            on = {}                               # one casualty
            on["position"] = ce
            on["shape"] = sh
            on["age_group"] = agn[sh]
            on["age_score"] = ags[sh]
            on["colour"] = cn
            on["severity"] = svn[cn]
            on["severity_score"] = svs[cn]
            on["priority"] = ags[sh] * svs[cn]
            on["elevation"] = find_elevation(ce, gm)
            cl.append(on)

    print("")
    print("2. CASUALTY INFORMATION")
    print("Total number of casualties =", len(cl))
    op = []                                       # only positions
    for c in cl:
        op.append(c["position"])
    print("Casualty coordinates =", op)
    nb = 1                                        # number
    for c in cl:
        print("   Casualty", nb, ":", c["position"],
              " shape =", c["shape"],
              " age group =", c["age_group"],
              " severity =", c["severity"],
              " priority =", c["priority"],
              " elevation level =", c["elevation"])
        nb = nb + 1

    # ---- path ----
    rv = MapForRover(cg, lm)                      # rover map

    # all the important points , "S" = start , "E" = end , 0,1,2.. = casualties
    ap = {}                                       # all points
    ap["S"] = rv.point_to_cell(sp)
    ap["E"] = rv.point_to_cell(ep)
    for i in range(len(cl)):
        ap[i] = rv.point_to_cell(cl[i]["position"])

    # find the path between every two points and keep them saved
    svp = {}                                      # saved paths
    ky = list(ap.keys())                          # keys
    for a in ky:
        for b in ky:
            if a == b:
                continue
            if (b, a) in svp:
                # I already did the opposite one so I just reverse it
                p, d, t = svp[(b, a)]
                if p is None:
                    svp[(a, b)] = (None, d, t)
                else:
                    svp[(a, b)] = (list(reversed(p)), d, t)
                continue
            svp[(a, b)] = rv.find_path(ap[a], ap[b])

    # ---- deciding in which order to visit the casualties ----
    # score = (displacement from start / distance travelled) * priority
    # the distance travelled keeps increasing so the casualties I visit later
    # give me a very small score. So the order is very important.
    # Trying every order is n! which is too slow , so I do it greedily :
    # every time I go to the casualty which gives me the biggest score right now.
    lv = list(range(len(cl)))                     # left to visit
    vo = []                                       # visiting order
    wa = "S"                                      # where i am
    disnw = 0.0                                   # distance till now

    while len(lv) > 0:
        bo = None                                 # best one
        bsc = -1                                  # best score
        for i in lv:
            p, d, t = svp[(wa, i)]
            if p is None:
                continue                          # cannot reach it (maybe behind water)
            tdd = disnw + d                       # total distance
            if tdd <= 0:
                continue
            dp = math.dist(sp, cl[i]["position"])         # displacement
            sc = (dp / tdd) * cl[i]["priority"]           # score
            if sc > bsc:
                bsc = sc
                bo = i
        if bo is None:
            break                                 # nothing else can be reached
        p, d, t = svp[(wa, bo)]
        disnw = disnw + d
        vo.append(bo)
        lv.remove(bo)
        wa = bo

    # now make the real path and count the score
    fl = []                                       # full path
    disnw = 0.0
    tt = 0.0                                      # total time
    asc = []                                      # all scores
    wa = "S"

    for i in vo:
        p, d, t = svp[(wa, i)]
        if p is None:
            continue
        pts = []                                  # points
        for ce in p:
            pts.append(rv.cell_to_point(ce))
        if len(fl) == 0:
            fl = fl + pts
        else:
            fl = fl + pts[1:]                     # skip repeat of the joining point

        disnw = disnw + d
        tt = tt + t
        dp = math.dist(sp, cl[i]["position"])
        sc = (dp / disnw) * cl[i]["priority"]

        cl[i]["displacement"] = dp
        cl[i]["travelled"] = disnw
        cl[i]["score"] = sc
        asc.append(sc)
        wa = i

    # at the end go to the purple triangle as this is the finalk destinationn
    p, d, t = svp[(wa, "E")]
    if p is not None:
        pts = []
        for ce in p:
            pts.append(rv.cell_to_point(ce))
        if len(fl) == 0:
            fl = fl + pts
        else:
            fl = fl + pts[1:]
        disnw = disnw + d
        tt = tt + t

    print("")
    print("3. ROVER PATH")
    print("Number of points in the path =", len(fl))
    print("First 10 points =", fl[:10], "....")
    print("Last 5 points   =", fl[-5:])
    # the full path has thousands of points so I save it in a text file
    f = open(of + "/" + nm + "_path.txt", "w")
    f.write(str(fl))
    f.close()
    print("(full path saved in", nm + "_path.txt )")

    print("")
    print("4. PATH SCORE")
    ts = 0.0                                      # total score
    nb = 1
    for i in vo:
        c = cl[i]
        if "score" not in c:
            continue
        ts = ts + c["score"]
        print("   " + str(nb) + ") " + str(c["position"]) +
              " | age = " + c["age_group"] +
              " | severity = " + c["severity"] +
              " | priority = " + str(c["priority"]) +
              " | displacement = " + str(round(c["displacement"], 2)) + " px" +
              " | travelled = " + str(round(c["travelled"], 2)) + " px" +
              " | score = " + str(round(c["score"], 3)))
        nb = nb + 1

    rnd = []                                      # rounded
    for s in asc:
        rnd.append(round(s, 3))
    print("   Casualty Scores =", rnd)
    print("   TOTAL PATH SCORE =", round(ts, 3))

    print("")
    print("6. TIME")
    print("   Total distance travelled =", round(disnw, 2), "px")
    print("   TOTAL TIME =", round(tt, 2), "seconds")

    # OUTPUT 5 : draw the path on the image
    dw = im.copy()                                # drawing
    for i in range(1, len(fl)):
        cv2.line(dw, fl[i - 1], fl[i], (255, 0, 255), 3)

    for c in cl:
        if "score" in c:
            rc = (0, 0, 255)      # visited
        else:
            rc = (128, 128, 128)  # could not reach
        cv2.circle(dw, c["position"], 16, rc, 2)
        cv2.putText(dw, str(c["priority"]),
                    (c["position"][0] + 18, c["position"][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(dw, str(c["priority"]),
                    (c["position"][0] + 18, c["position"][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.circle(dw, sp, 12, (0, 0, 0), -1)
    cv2.circle(dw, sp, 9, (255, 255, 255), -1)
    cv2.putText(dw, "START", (sp[0] - 25, sp[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.circle(dw, ep, 12, (0, 0, 0), -1)
    cv2.circle(dw, ep, 9, (0, 255, 255), -1)
    cv2.putText(dw, "END", (ep[0] - 18, ep[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(of + "/" + nm + "_5_path.png", dw)

    rs = {}                                       # result
    rs["name"] = nm
    rs["score"] = ts
    rs["time"] = tt
    return rs


# main , runs on every image in the folder

def main():
    # I use the folder where THIS FILE is kept, not the folder from where I run
    # the command, otherwise the outputs folder gets made in the wrong place.
    hr = os.path.dirname(os.path.abspath(__file__))     # here
    inf = os.path.join(hr, "images")                    # input folder
    of = os.path.join(hr, "outputs")                    # output folder
    if not os.path.exists(of):
        os.makedirs(of)

    fn = []                                       # file names
    for f in sorted(os.listdir(inf)):
        if f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
            fn.append(f)

    ar = []                                       # all results
    for f in fn:
        r = do_one_image(inf + "/" + f, of)
        if r is not None:
            ar.append(r)

    print("")
    print("==================================================================")
    print("7. GLOBAL RANKING")
    print("==================================================================")

    bsc = sorted(ar, key=lambda r: -r["score"])   # by score
    bt = sorted(ar, key=lambda r: r["time"])      # by time

    n1 = []
    for r in bsc:
        n1.append(r["name"])
    print("Path Score Ranking (highest first) =", n1)
    for r in bsc:
        print("   ", r["name"], " score =", round(r["score"], 3))

    n2 = []
    for r in bt:
        n2.append(r["name"])
    print("Time Ranking (fastest first) =", n2)
    for r in bt:
        print("   ", r["name"], " time =", round(r["time"], 2), "s")


main()
