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

import cv2
import numpy as np
import os
from collections import deque


step_size = 20        # the circle moves 20 pixels in one step
grid = 10             # I plan on a small grid of 10 px to make it fast


# -------------------------------------------------------------------------
# PART 1 : the only time I am allowed to look at the full image
# -------------------------------------------------------------------------

def find_start_and_target(image):
    # the image has noise in it so I use blur first, otherwise the mask
    # comes out full of small dots
    blur = cv2.medianBlur(image, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # white circle = i am here. white means saturation very low, value very high
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    # black circle = my target. everything dark
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))

    # clean the small dots
    kernel = np.ones((5, 5), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)

    def biggest_centre(mask):
        contours, junk = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, 0
        big = max(contours, key=cv2.contourArea)
        m = cv2.moments(big)
        if m["m00"] == 0:
            return None, 0
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        radius = int(np.sqrt(cv2.contourArea(big) / np.pi))
        return (x, y), radius

    start, my_radius = biggest_centre(white)
    target, junk = biggest_centre(black)
    return start, target, my_radius


# -------------------------------------------------------------------------
# PART 2 : this is the ONLY thing I can look at while walking
# -------------------------------------------------------------------------

def get_receptive_field(image, centre, rf_w, rf_h):
    """
    Cuts out a small window around me. This is the only part of the world
    I am allowed to see. It also tells me where this window sits, because
    I need that to change the small coordinates into big coordinates.
    """
    h, w = image.shape[:2]
    x1 = centre[0] - rf_w // 2
    y1 = centre[1] - rf_h // 2
    # if the window goes outside the image I push it back inside
    x1 = max(0, min(x1, w - rf_w))
    y1 = max(0, min(y1, h - rf_h))
    crop = image[y1:y1 + rf_h, x1:x1 + rf_w]
    return crop, x1, y1


def free_space_in_crop(crop):
    """
    In the crop, green = I can walk, red = wall.
    The image is noisy so I blur it first.
    I take everything that is NOT red as walkable (green, and also the
    white and black circles are walkable).
    """
    blur = cv2.medianBlur(crop, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 90, 90]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red, red2)
    # close the small holes made by the noise so the wall is one solid block
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free = cv2.bitwise_not(red)
    return free


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


def remember(memory, free_mask, x1, y1, safe):
    """paste what I just saw into my memory map"""
    h, w = free_mask.shape
    # I make the walls a bit fatter by the radius of my circle, so that the
    # centre of my circle never goes so close that my body touches the wall
    fat_wall = cv2.dilate((free_mask == 0).astype(np.uint8),
                          np.ones((safe * 2 + 1, safe * 2 + 1), np.uint8))
    part = memory[y1:y1 + h, x1:x1 + w]
    part[fat_wall > 0] = WALL
    part[(fat_wall == 0)] = FREE


def plan_full_path(memory, me, target):
    """
    BFS on my memory map from where I am to the target.
    The places I have not seen yet are counted as FREE (I am hopeful there is
    a way there). If I go there and find a wall, my memory gets corrected and
    I plan again. This is how a real robot explores.
    Gives back the full list of grid cells to follow.
    """
    h, w = memory.shape
    gh, gw = h // grid, w // grid

    def cell_is_ok(r, c):
        if r < 0 or c < 0 or r >= gh or c >= gw:
            return False
        piece = memory[r * grid:(r + 1) * grid, c * grid:(c + 1) * grid]
        return not (piece == WALL).any()

    start_cell = (me[1] // grid, me[0] // grid)
    goal_cell = (target[1] // grid, target[0] // grid)

    came_from = {start_cell: None}
    queue = deque([start_cell])
    found = False
    while len(queue) > 0:
        cur = queue.popleft()
        if cur == goal_cell:
            found = True
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in came_from and cell_is_ok(nxt[0], nxt[1]):
                came_from[nxt] = cur
                queue.append(nxt)

    if not found:
        return None

    path = []
    cur = goal_cell
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def path_is_still_ok(memory, path):
    """check that the next few cells of my plan are not walls now"""
    if path is None or len(path) < 2:
        return False
    gh, gw = memory.shape[0] // grid, memory.shape[1] // grid
    for cell in path[1:6]:              # I only check a few cells ahead
        r, c = cell
        if r < 0 or c < 0 or r >= gh or c >= gw:
            return False
        piece = memory[r * grid:(r + 1) * grid, c * grid:(c + 1) * grid]
        if (piece == WALL).any():
            return False
    return True


def direction_to_cell(me, cell):
    """
    I can only move up / down / left / right, 20 px at a time.
    So to reach the next cell I first fix the x, then the y.
    (if I want to go diagonally I need two steps, like the pdf says)
    """
    want_x = cell[1] * grid + grid // 2
    want_y = cell[0] * grid + grid // 2
    if abs(want_x - me[0]) >= step_size // 2:
        if want_x > me[0]:
            return (step_size, 0)
        else:
            return (-step_size, 0)
    if abs(want_y - me[1]) >= step_size // 2:
        if want_y > me[1]:
            return (0, step_size)
        else:
            return (0, -step_size)
    return None                          # I am already on this cell


def can_i_stand_here(memory, point, w, h):
    """check my memory : is this new place free ?"""
    x, y = point
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return memory[y, x] != WALL


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def run(image_path, out_folder, max_steps=400):
    name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # the receptive field is (width/4 x height/2) like the pdf says
    rf_w = width // 4
    rf_h = height // 2

    print("==================================================")
    print("IMAGE :", name, " size =", width, "x", height)
    print("Receptive field size =", rf_w, "x", rf_h)

    # ---- the one and only global look ----
    me, target, my_radius = find_start_and_target(image)
    if me is None or target is None:
        print("could not find the white circle or the black circle")
        return
    print("White circle (start) =", me)
    print("Black circle (target)=", target)
    print("my radius =", my_radius)
    print("--- from here I can only see my receptive field ---")

    snap_folder = out_folder + "/" + name + "_steps"
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)

    # my memory of the world, everything unknown in the beginning
    memory = np.zeros((height, width), np.uint8)

    walked = [me]
    frames = []
    my_plan = None
    plan_index = 1
    stuck_count = 0

    for step in range(max_steps):
        # STEP A : look around me (this is all I am allowed to see)
        crop, x1, y1 = get_receptive_field(image, me, rf_w, rf_h)
        free = free_space_in_crop(crop)

        # STEP B : remember what I saw
        remember(memory, free, x1, y1, my_radius + 2)

        # STEP C : save the snapshot of this moment
        frame = draw_snapshot(image, me, target, walked, x1, y1, rf_w, rf_h,
                              my_radius, step, memory)
        cv2.imwrite(snap_folder + "/step_%04d.png" % step, frame)
        frames.append(frame)

        # have I reached ?
        if abs(me[0] - target[0]) <= step_size and abs(me[1] - target[1]) <= step_size:
            print("REACHED the black circle in", step, "steps")
            break

        # STEP D : if I have no plan, or my plan is blocked now, plan again
        if not path_is_still_ok(memory, my_plan):
            my_plan = plan_full_path(memory, me, target)
            plan_index = 1
            if my_plan is None:
                stuck_count = stuck_count + 1
                print("step", step, ": no way found in my memory")
                if stuck_count > 5:
                    print("I am stuck, stopping")
                    break
                continue
            stuck_count = 0

        # skip the cells I am already standing on
        move = None
        while plan_index < len(my_plan):
            move = direction_to_cell(me, my_plan[plan_index])
            if move is not None:
                break
            plan_index = plan_index + 1
        if move is None:
            my_plan = None
            continue

        new_place = (me[0] + move[0], me[1] + move[1])

        # STEP E : do not walk into a wall
        if not can_i_stand_here(memory, new_place, width, height):
            memory[max(0, new_place[1] - 3):new_place[1] + 4,
                   max(0, new_place[0] - 3):new_place[0] + 4] = WALL
            my_plan = None              # my plan was wrong, make a new one
            continue

        me = new_place
        walked.append(me)

    print("total steps walked =", len(walked) - 1)
    print("snapshots saved in", snap_folder)

    # ---- make the video ----
    video_path = out_folder + "/" + name + "_video.mp4"
    if len(frames) > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
        for f in frames:
            video.write(f)
        # keep the last frame for a moment so we can see it reached
        for i in range(15):
            video.write(frames[-1])
        video.release()
        print("video saved in", video_path)

    # also save the final picture with the full path drawn
    final = image.copy()
    for i in range(1, len(walked)):
        cv2.line(final, walked[i - 1], walked[i], (255, 0, 255), 3)
    cv2.circle(final, walked[0], my_radius, (255, 255, 255), -1)
    cv2.circle(final, walked[0], my_radius, (0, 0, 0), 2)
    cv2.circle(final, target, 12, (0, 255, 255), 3)
    cv2.imwrite(out_folder + "/" + name + "_final_path.png", final)
    print("final path image saved")


def draw_snapshot(image, me, target, walked, x1, y1, rf_w, rf_h, radius, step, memory):
    """
    The snapshot shows what the robot can see. Everything outside the
    receptive field is made dark, because the robot cannot see it.
    """
    frame = (image * 0.25).astype(np.uint8)          # dark = I cannot see this
    frame[y1:y1 + rf_h, x1:x1 + rf_w] = image[y1:y1 + rf_h, x1:x1 + rf_w]

    # yellow box = my receptive field
    cv2.rectangle(frame, (x1, y1), (x1 + rf_w, y1 + rf_h), (0, 255, 255), 3)

    # the path I have walked till now
    for i in range(1, len(walked)):
        cv2.line(frame, walked[i - 1], walked[i], (255, 0, 255), 3)

    # me
    cv2.circle(frame, me, radius, (255, 255, 255), -1)
    cv2.circle(frame, me, radius, (0, 0, 0), 2)

    # writing on top
    cv2.putText(frame, "step " + str(step), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, "step " + str(step), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "receptive field " + str(rf_w) + "x" + str(rf_h), (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, "receptive field " + str(rf_w) + "x" + str(rf_h), (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return frame


def main():
    # I have only one maze image to work on, so I give the file name directly
    # instead of looping over a folder.
    input_image = "task2\images\maze1.jpg"
    output_folder = "task2\outputs"

    run(input_image, output_folder)


if __name__ == "__main__":
    main()
