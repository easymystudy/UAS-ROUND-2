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
age_score_of = {"circle": 3, "star": 1, "square": 2}
age_name_of = {"circle": "Children", "star": "Adults", "square": "Senior Citizens"}

#colour scores
severity_score_of = {"red": 3, "yellow": 2, "white": 1}
severity_name_of = {"red": "Critical", "yellow": "Moderate", "white": "Safe"}

# speed of rover on each green level
# level 0 = light green (grounded ) , level 1 = medium green (above the ground) , level 2 = darkest green( at top levl)
speed_list = [20.0, 15.0, 10.0]

#setting the pixel to which it will see we can adjust it, if i want the route in less time we will go with large value and most accurate with 1
box_size = 1


# STEP 1 : finding the different colours

def get_hsv(image):
    # hsv is easier to use than bgr for finding colours
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def find_colour(image, low1, high1, low2=None, high2=None):
    # this makes a black and white mask , white where the colour is found
    hsv = get_hsv(image)
    mask = cv2.inRange(hsv, np.array(low1), np.array(high1))
    if low2 is not None:
        mask2 = cv2.inRange(hsv, np.array(low2), np.array(high2))
        mask = cv2.bitwise_or(mask, mask2)
    # remove the very small dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def find_black(image):
    # black region = obstacle , rover cannot pass
    hsv = get_hsv(image)
    return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))


def find_water(image):
    # the big blue oval is a water body so the rover CANNOT go there.
    # I checked the colour of the water with print(img[y,x]) and got
    # hsv about (130, 230, 235) which is a very strong blue.
    hsv = get_hsv(image)
    water = cv2.inRange(hsv, np.array([100, 180, 60]), np.array([140, 255, 255]))
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return water


def find_all_greens(image):
    # this gives me a list of green masks
    # first one is the lightest green (level 0) then darker and darker
    hsv = get_hsv(image)
    green = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([90, 255, 255]))

    green_pixels = image[green > 0]
    if len(green_pixels) == 0:
        return [np.zeros(green.shape, np.uint8)]

    # First I tried to separate the 3 greens using brightness ranges but when
    # I printed np.unique() I got hundreds of values because the edges of the
    # ellipses are blurred (anti aliasing). So now I find the main flat green
    # colours and then give every green pixel to the closest main colour.
    colours, how_many = np.unique(green_pixels.reshape(-1, 3), axis=0, return_counts=True)
    big_enough = how_many > 0.01 * len(green_pixels)
    main_colours = colours[big_enough].astype(np.int32)
    main_counts = how_many[big_enough]

    # Now which green is level 0 , level 1 , level 2 ?
    # The elevations are drawn like circles inside circles , so the ground
    # covers the biggest area and the higher levels cover smaller areas.
    # So I sort by area , biggest area = level 0.
    main_colours = main_colours[np.argsort(-main_counts)]

    # for every green pixel find which main colour is nearest
    difference = green_pixels.astype(np.int32)[:, None, :] - main_colours[None, :, :]
    nearest = np.argmin((difference ** 2).sum(axis=2), axis=1)

    all_y, all_x = np.nonzero(green)
    green_masks = []
    for i in range(len(main_colours)):
        one_mask = np.zeros(green.shape, np.uint8)
        picked = (nearest == i)
        one_mask[all_y[picked], all_x[picked]] = 255
        green_masks.append(one_mask)
    return green_masks

# STEP 2 : finding what shape it is

def what_shape(one_contour):
    # circle = child , star = adult , square = senior , triangle = start/end marker
    length = cv2.arcLength(one_contour, True)
    corners = len(cv2.approxPolyDP(one_contour, 0.02 * length, True))
    area = cv2.contourArea(one_contour)
    if area < 1:
        return "unknown"

    # a star is very spiky so a lot of its outer box is empty.
    # solidity = area / area of the convex hull
    hull_area = cv2.contourArea(cv2.convexHull(one_contour))
    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 1.0

    if solidity < 0.75 or corners >= 9:
        return "star"
    if corners == 3:
        return "triangle"
    if corners == 4:
        return "square"

    # roundness = 4 * pi * area / (perimeter * perimeter) , it is 1 for a circle  
    if length > 0:
        roundness = 4 * math.pi * area / (length * length)
    else:
        roundness = 0
    if roundness > 0.7:
        return "circle"
    return "square"


def find_shapes_in_mask(mask, smallest_area):
    # returns a list like [(shape name, (x,y)), ...]
    contours, junk = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer = []
    for c in contours:
        if cv2.contourArea(c) < smallest_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        centre_x = int(m["m10"] / m["m00"])
        centre_y = int(m["m01"] / m["m00"])
        answer.append((what_shape(c), (centre_x, centre_y)))
    return answer


def find_elevation(point, green_masks, size=14):
    # the casualty shape is drawn ON the green so I cannot see the green under it.
    # So I look at a small square around the shape and see which green is most there.
    x, y = point
    height, width = green_masks[0].shape
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(width, x + size)
    y2 = min(height, y + size)

    counts = []
    for m in green_masks:
        counts.append(int(np.count_nonzero(m[y1:y2, x1:x2])))

    if sum(counts) == 0:
        if size < 100:
            return find_elevation(point, green_masks, size * 3)   # look in a bigger square
        return 0
    return int(np.argmax(counts))


# STEP 3 : path finding with A star

def get_speed(level):
    if level < len(speed_list):
        return speed_list[level]
    return speed_list[-1]      # if there are more than 3 greens use slowest


class MapForRover:
    # I make a smaller grid of the image so that A star is fast

    def __init__(self, can_go_mask, level_map):
        self.free = can_go_mask[::box_size, ::box_size] > 0
        self.level = level_map[::box_size, ::box_size]
        self.rows, self.cols = self.free.shape

    def point_to_cell(self, point):
        return (point[1] // box_size, point[0] // box_size)   # (row, col)

    def cell_to_point(self, cell):
        return (cell[1] * box_size, cell[0] * box_size)       # (x, y)

    def is_free(self, cell):
        r, c = cell
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return False
        return self.free[r, c]

    def closest_free_cell(self, cell):
        # if the point is on an obstacle or on a shape , move it to the nearest free cell
        if self.is_free(cell):
            return cell
        for ring in range(1, 80):
            for d in range(-ring, ring + 1):
                for side in (-ring, ring):
                    try1 = (cell[0] + d, cell[1] + side)
                    try2 = (cell[0] + side, cell[1] + d)
                    if self.is_free(try1):
                        return try1
                    if self.is_free(try2):
                        return try2
        return cell

    def find_path(self, start_cell, end_cell):
        # normal A star , returns (list of cells, distance in px, time in sec)
        start_cell = self.closest_free_cell(start_cell)
        end_cell = self.closest_free_cell(end_cell)
        if start_cell == end_cell:
            return [start_cell], 0.0, 0.0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        to_check = [(0.0, start_cell)]
        came_from = {}
        cost_so_far = {start_cell: 0.0}
        already_done = set()

        while len(to_check) > 0:
            junk, now = heapq.heappop(to_check)
            if now in already_done:
                continue
            already_done.add(now)
            if now == end_cell:
                break

            for dr, dc in moves:
                nxt = (now[0] + dr, now[1] + dc)
                if not self.is_free(nxt):
                    continue
                step_dist = math.hypot(dr, dc) * box_size
                new_cost = cost_so_far[now] + step_dist
                if new_cost < cost_so_far.get(nxt, 999999999):
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = now
                    guess = math.hypot(nxt[0] - end_cell[0], nxt[1] - end_cell[1]) * box_size
                    heapq.heappush(to_check, (new_cost + guess, nxt))

        if end_cell not in came_from:
            return None, float("inf"), float("inf")    # no path found

        # go backwards to make the path
        path = [end_cell]
        while path[-1] != start_cell:
            path.append(came_from[path[-1]])
        path.reverse()

        total_dist = 0.0
        total_time = 0.0
        for i in range(1, len(path)):
            a = path[i - 1]
            b = path[i]
            d = math.hypot(a[0] - b[0], a[1] - b[1]) * box_size
            total_dist = total_dist + d
            total_time = total_time + d / get_speed(int(self.level[b[0], b[1]]))
        return path, total_dist, total_time


# doing everything for one image

def do_one_image(image_path, output_folder):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print("cannot open", image_path)
        return None

    height, width = image.shape[:2]
    print("")
    print("==================================================================")
    print("IMAGE :", image_name, "  size =", width, "x", height)
    print("==================================================================")

    # ---- masks ----
    green_masks = find_all_greens(image)
    black_mask = find_black(image)
    water_mask = find_water(image)
    print("number of green shades found =", len(green_masks))

    all_green = np.zeros((height, width), np.uint8)
    for m in green_masks:
        all_green = cv2.bitwise_or(all_green, m)

    # the shapes are drawn on top of the green so those pixels are not green,
    # but the rover can still stand there , so I add them back
    red_mask = find_colour(image, [0, 120, 90], [8, 255, 255], [170, 120, 90], [180, 255, 255])
    yellow_mask = find_colour(image, [20, 90, 150], [34, 255, 255])
    white_mask = find_colour(image, [0, 0, 200], [180, 40, 255])
    orange_mask = find_colour(image, [9, 120, 120], [19, 255, 255])
    purple_mask = find_colour(image, [125, 60, 120], [160, 179, 255])

    shapes_mask = np.zeros((height, width), np.uint8)
    for m in [red_mask, yellow_mask, white_mask, orange_mask, purple_mask]:
        shapes_mask = cv2.bitwise_or(shapes_mask, m)

    can_go = cv2.bitwise_or(all_green, shapes_mask)
    can_go[black_mask > 0] = 0      # black obstacle
    can_go[water_mask > 0] = 0      # water body , rover cannot swim
    can_go = cv2.morphologyEx(can_go, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    can_go[black_mask > 0] = 0      # doing it again so the closing does not eat the wall
    can_go[water_mask > 0] = 0

    cannot_go = cv2.bitwise_not(can_go)

    # OUTPUT 1 : the mask image
    mask_picture = np.zeros((height, width, 3), np.uint8)
    mask_picture[can_go > 0] = (255, 255, 255)
    cv2.imwrite(output_folder + "/" + image_name + "_1_mask.png", mask_picture)

    # a colour version so it is easy to see what is what
    colour_picture = np.zeros((height, width, 3), np.uint8)
    nice_greens = [(120, 255, 120), (60, 180, 60), (20, 100, 20)]
    for i in range(len(green_masks)):
        colour_picture[green_masks[i] > 0] = nice_greens[min(i, 2)]
    colour_picture[black_mask > 0] = (0, 0, 255)        # red = wall
    colour_picture[water_mask > 0] = (255, 100, 0)      # blue = water
    cv2.imwrite(output_folder + "/" + image_name + "_1b_terrain.png", colour_picture)

    # this map tells which green level every pixel is on , needed for the speed
    level_map = np.zeros((height, width), np.uint8)
    for i in range(len(green_masks)):
        level_map[green_masks[i] > 0] = i

    # ---- start and end ----
    # NOTE : there is also an orange CIRCLE casualty in the image, so I must take
    # only the TRIANGLE otherwise the start point comes out wrong.
    start_point = None
    end_point = None
    for shape, centre in find_shapes_in_mask(orange_mask, 200):
        if shape == "triangle":
            start_point = centre
    for shape, centre in find_shapes_in_mask(purple_mask, 200):
        if shape == "triangle":
            end_point = centre

    if start_point is None or end_point is None:
        print("could not find the start or the end triangle")
        return None

    print("Start (orange triangle)      =", start_point)
    print("Destination (purple triangle)=", end_point)

    # ---- casualties ----
    casualty_list = []
    for colour_name, colour_mask in [("red", red_mask), ("yellow", yellow_mask), ("white", white_mask)]:
        for shape, centre in find_shapes_in_mask(colour_mask, 120):
            if shape not in age_score_of:
                continue         # triangle or unknown , it is not a casualty
            one = {}
            one["position"] = centre
            one["shape"] = shape
            one["age_group"] = age_name_of[shape]
            one["age_score"] = age_score_of[shape]
            one["colour"] = colour_name
            one["severity"] = severity_name_of[colour_name]
            one["severity_score"] = severity_score_of[colour_name]
            one["priority"] = age_score_of[shape] * severity_score_of[colour_name]
            one["elevation"] = find_elevation(centre, green_masks)
            casualty_list.append(one)

    print("")
    print("2. CASUALTY INFORMATION")
    print("Total number of casualties =", len(casualty_list))
    only_positions = []
    for c in casualty_list:
        only_positions.append(c["position"])
    print("Casualty coordinates =", only_positions)
    number = 1
    for c in casualty_list:
        print("   Casualty", number, ":", c["position"],
              " shape =", c["shape"],
              " age group =", c["age_group"],
              " severity =", c["severity"],
              " priority =", c["priority"],
              " elevation level =", c["elevation"])
        number = number + 1

    # ---- path ----
    rover_map = MapForRover(can_go, level_map)

    # all the important points , "S" = start , "E" = end , 0,1,2.. = casualties
    all_points = {}
    all_points["S"] = rover_map.point_to_cell(start_point)
    all_points["E"] = rover_map.point_to_cell(end_point)
    for i in range(len(casualty_list)):
        all_points[i] = rover_map.point_to_cell(casualty_list[i]["position"])

    # find the path between every two points and keep them saved
    saved_paths = {}
    keys = list(all_points.keys())
    for a in keys:
        for b in keys:
            if a == b:
                continue
            if (b, a) in saved_paths:
                # I already did the opposite one so I just reverse it
                p, d, t = saved_paths[(b, a)]
                if p is None:
                    saved_paths[(a, b)] = (None, d, t)
                else:
                    saved_paths[(a, b)] = (list(reversed(p)), d, t)
                continue
            saved_paths[(a, b)] = rover_map.find_path(all_points[a], all_points[b])

    # ---- deciding in which order to visit the casualties ----
    # score = (displacement from start / distance travelled) * priority
    # the distance travelled keeps increasing so the casualties I visit later
    # give me a very small score. So the order is very important.
    # Trying every order is n! which is too slow , so I do it greedily :
    # every time I go to the casualty which gives me the biggest score right now.
    left_to_visit = list(range(len(casualty_list)))
    visiting_order = []
    where_i_am = "S"
    distance_till_now = 0.0

    while len(left_to_visit) > 0:
        best_one = None
        best_score = -1
        for i in left_to_visit:
            p, d, t = saved_paths[(where_i_am, i)]
            if p is None:
                continue                      # cannot reach it (maybe behind water)
            total_d = distance_till_now + d
            if total_d <= 0:
                continue
            displacement = math.dist(start_point, casualty_list[i]["position"])
            score = (displacement / total_d) * casualty_list[i]["priority"]
            if score > best_score:
                best_score = score
                best_one = i
        if best_one is None:
            break                             # nothing else can be reached
        p, d, t = saved_paths[(where_i_am, best_one)]
        distance_till_now = distance_till_now + d
        visiting_order.append(best_one)
        left_to_visit.remove(best_one)
        where_i_am = best_one

    # now make the real path and count the score 
    full_path = []
    distance_till_now = 0.0
    total_time = 0.0
    all_scores = []
    where_i_am = "S"

    for i in visiting_order:
        p, d, t = saved_paths[(where_i_am, i)]
        if p is None:
            continue
        points = []
        for cell in p:
            points.append(rover_map.cell_to_point(cell))
        if len(full_path) == 0:
            full_path = full_path + points
        else:
            full_path = full_path + points[1:]     # skip repeat of the joining point

        distance_till_now = distance_till_now + d
        total_time = total_time + t
        displacement = math.dist(start_point, casualty_list[i]["position"])
        score = (displacement / distance_till_now) * casualty_list[i]["priority"]

        casualty_list[i]["displacement"] = displacement
        casualty_list[i]["travelled"] = distance_till_now
        casualty_list[i]["score"] = score
        all_scores.append(score)
        where_i_am = i

    # at the end go to the purple triangle as this is the finalk destinationn
    p, d, t = saved_paths[(where_i_am, "E")]
    if p is not None:
        points = []
        for cell in p:
            points.append(rover_map.cell_to_point(cell))
        if len(full_path) == 0:
            full_path = full_path + points
        else:
            full_path = full_path + points[1:]
        distance_till_now = distance_till_now + d
        total_time = total_time + t

    print("")
    print("3. ROVER PATH")
    print("Number of points in the path =", len(full_path))
    print("First 10 points =", full_path[:10], "....")
    print("Last 5 points   =", full_path[-5:])
    # the full path has thousands of points so I save it in a text file
    f = open(output_folder + "/" + image_name + "_path.txt", "w")
    f.write(str(full_path))
    f.close()
    print("(full path saved in", image_name + "_path.txt )")

    print("")
    print("4. PATH SCORE")
    total_score = 0.0
    number = 1
    for i in visiting_order:
        c = casualty_list[i]
        if "score" not in c:
            continue
        total_score = total_score + c["score"]
        print("   " + str(number) + ") " + str(c["position"]) +
              " | age = " + c["age_group"] +
              " | severity = " + c["severity"] +
              " | priority = " + str(c["priority"]) +
              " | displacement = " + str(round(c["displacement"], 2)) + " px" +
              " | travelled = " + str(round(c["travelled"], 2)) + " px" +
              " | score = " + str(round(c["score"], 3)))
        number = number + 1

    rounded = []
    for s in all_scores:
        rounded.append(round(s, 3))
    print("   Casualty Scores =", rounded)
    print("   TOTAL PATH SCORE =", round(total_score, 3))

    print("")
    print("6. TIME")
    print("   Total distance travelled =", round(distance_till_now, 2), "px")
    print("   TOTAL TIME =", round(total_time, 2), "seconds")

    # OUTPUT 5 : draw the path on the image 
    drawing = image.copy()
    for i in range(1, len(full_path)):
        cv2.line(drawing, full_path[i - 1], full_path[i], (255, 0, 255), 3)

    for c in casualty_list:
        if "score" in c:
            ring_colour = (0, 0, 255)      # visited
        else:
            ring_colour = (128, 128, 128)  # could not reach
        cv2.circle(drawing, c["position"], 16, ring_colour, 2)
        cv2.putText(drawing, str(c["priority"]),
                    (c["position"][0] + 18, c["position"][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(drawing, str(c["priority"]),
                    (c["position"][0] + 18, c["position"][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.circle(drawing, start_point, 12, (0, 0, 0), -1)
    cv2.circle(drawing, start_point, 9, (255, 255, 255), -1)
    cv2.putText(drawing, "START", (start_point[0] - 25, start_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.circle(drawing, end_point, 12, (0, 0, 0), -1)
    cv2.circle(drawing, end_point, 9, (0, 255, 255), -1)
    cv2.putText(drawing, "END", (end_point[0] - 18, end_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(output_folder + "/" + image_name + "_5_path.png", drawing)

    result = {}
    result["name"] = image_name
    result["score"] = total_score
    result["time"] = total_time
    return result


# main , runs on every image in the folder

def main():
    input_folder = "images"
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_names = []
    for f in sorted(os.listdir(input_folder)):
        if f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
            file_names.append(f)

    all_results = []
    for f in file_names:
        r = do_one_image(input_folder + "/" + f, output_folder)
        if r is not None:
            all_results.append(r)

    print("")
    print("==================================================================")
    print("7. GLOBAL RANKING")
    print("==================================================================")

    by_score = sorted(all_results, key=lambda r: -r["score"])
    by_time = sorted(all_results, key=lambda r: r["time"])

    names1 = []
    for r in by_score:
        names1.append(r["name"])
    print("Path Score Ranking (highest first) =", names1)
    for r in by_score:
        print("   ", r["name"], " score =", round(r["score"], 3))

    names2 = []
    for r in by_time:
        names2.append(r["name"])
    print("Time Ranking (fastest first) =", names2)
    for r in by_time:
        print("   ", r["name"], " time =", round(r["time"], 2), "s")


main()
