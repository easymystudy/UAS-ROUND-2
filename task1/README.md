# UAS-DTU Round 2 — Task 1
## Casualty Analysis using a Rover Guided by a UAV

My solution file is **`task1.py`**. I have used only **OpenCV, NumPy and normal python maths**.

All the output images are saved inside the `outputs/` folder.

---

## 1. What the question is asking (in simple words)

A drone flew over a disaster area and gave us **one top-view colour coded picture**. A rover
has to drive on the ground using this picture. Nothing is given to us in numbers, my program
has to **read the pixels with OpenCV** and find out everything by itself.

**Meaning of the colours:**
 Colour                                                    Meaning
 Black                    -                   Obstacle, rover **cannot** go 
 Blue oval                                    **Water body, rover cannot go** 
 Light green                                  Normal ground, elevation level 0, speed **20 px/s** 
 Medium dark green                            One level higher, level 1, speed **15 px/s** 
 Darkest green                                Highest level, level 2, speed **10 px/s** 
 Orange triangle                              Where the rover **starts** 
 Purple triangle                              Where the rover must **finish** (safe zone) 
 Other shapes                                 Casualties (injured people) 

**The casualties are told by their shape and colour:**

| Shape | Age Group | Age Score | | Colour | Condition | Severity Score |
|---|---|---|---|---|---|---|
| Circle | Children | 3 | | Red | Critical | 3 |
| Star | Adults | 1 | | Yellow | Moderate | 2 |
| Square | Senior Citizens | 2 | | White | Safe | 1 |

`Priority Score = Age Score x Severity Score`  (example : a red circle = 3 x 3 = 9)

### The scoring formula (this is the part which was confusing me)

```
Casualty Score = (Displacement from start / Distance travelled to reach it) x Priority
Total Path Score = sum of all the casualty scores
```

* **Displacement** = straight line distance from the orange triangle to that casualty. This
  never changes.
* **Distance travelled** = how much the rover has actually driven from the start till it
  reaches that casualty. This keeps **increasing** as the rover moves.

So the top part is fixed but the bottom part keeps getting bigger, which means the score
keeps getting smaller. **A casualty which I visit early gives a big score and a casualty**
**which I visit at the end gives almost nothing.**

The best idea is to first go to the casualties which have a **high priority** and which are
**far in a straight line** but still **cheap to reach**.

slope is being ignored as said by mentor in meeting and instructed in pdf

### What has to be printed for every image
1. Traversable / non traversable mask image
2. Number of casualties, their coordinates, age group, severity and elevation level
3. The rover path as a list of pixel coordinates
4. The path score table and the total score
5. The path drawn on the original image and saved
6. The total travel time
7. After all the images, rank them by score (big to small) and by time (small to big)

---

## 2. How my code works

### Step 1 : finding the colours
I change the image from BGR to **HSV** because in HSV it is much easier to say "this is
green" or "this is red" using `cv2.inRange()`.

* **Black obstacle** — the value channel is very low.
* **Water** — I checked the colour of the blue oval by printing `img[y,x]` and got HSV
  about `(130, 230, 235)`. The purple destination triangle is also blue-ish but its HSV is
  about `(143, 135, 230)`, its **saturation is much lower**. So I take only the pixels with
  saturation above 180, that way I get the water and **not** the purple triangle.
* **The three greens** — this was the hardest part, explained in the error analysis below.

The traversable mask = all the greens + all the coloured shapes, then I **remove the black
obstacles and the water**. The shapes have to be added because they are drawn on top of the
ground so those pixels are not green, but the rover can obviously stand there.

Saved as `*_1_mask.png` (white = can go) and `*_1b_terrain.png` (colour coded, red = wall,
blue = water, three greens = three levels).

### Step 2 : finding the casualties
For every colour (red / yellow / white) I make a mask and use `cv2.findContours`. Small
dots are ignored because they are just noise. Then I find out the shape :

* `cv2.approxPolyDP` gives me the number of corners → 3 = triangle, 4 = square.
* **Solidity** = contour area / convex hull area. A star is very spiky so a big part of its
  hull is empty, so solidity is small. This works much better than counting the 10 corners
  of a star.
* **Roundness** = 4πA/P², it is about 1 for a circle.
* Triangles are skipped because they are the start and end markers, not casualties.

The centre is found with `cv2.moments`.

For the **elevation**, the casualty shape is hiding the green under it, so I look at a small
square **around** the shape and see which green shade is present the most there.

### Step 3 : the path
Doing A* on every single pixel is too slow, so I make a smaller grid by taking every 4th
pixel (`box_size = 1`). Then I use normal **A\*** with 8 directions. A diagonal step counts
as `√2 x box_size`. If a point falls on an obstacle I move it to the nearest free cell.

I find the path between **every pair** of important points (start, every casualty, end) and
save them, so that I do not have to calculate them again and again.

### Step 4 : the order of visiting
Trying all the orders is `n!` which is way too slow. Because the formula punishes the
distance travelled so much, I do it **greedily** : every time I check which casualty would
give me the biggest score **right now**, and I go there. This automatically picks the high
priority far casualties first, which is exactly what the formula wants.

If a casualty cannot be reached at all (for example if it is on an island inside the water)
then A* returns nothing and my code just skips it and marks it with a grey circle.

### Step 5 : the time
For every small piece of the path I check which green level that pixel is on and do
`time = time + piece_length / speed_of_that_level` where the speed is 20, 15 or 10 px/s.

### Step 6 : the drawing
The path is drawn in pink on the original image, the casualties are circled with their
priority number written next to them, and the start and end are marked. Saved as
`*_5_path.png`. The full coordinate list has more than a thousand points so printing it
would fill the whole terminal, that is why I save it in `*_path.txt`.

### Step 7 : the ranking
After all the images are finished, the results are sorted two times and printed.

---

## 3. Output on the sample image

```
number of green shades found = 4
Start (orange triangle)      = (356, 610)
Destination (purple triangle)= (541, 69)

2. CASUALTY INFORMATION
Total number of casualties = 9
   (120, 586)  star    Adults           Critical  priority 3  elevation 0
   (813, 464)  star    Adults           Critical  priority 3  elevation 0
   (916, 118)  square  Senior Citizens  Critical  priority 6  elevation 2
   (41, 33)    square  Senior Citizens  Critical  priority 6  elevation 0
   (541, 435)  star    Adults           Moderate  priority 2  elevation 0
   (637, 154)  circle  Children         Moderate  priority 6  elevation 0
   (217, 457)  square  Senior Citizens  Safe      priority 2  elevation 0
   (916, 338)  circle  Children         Safe      priority 3  elevation 0
   (1155, 145) star    Adults           Safe      priority 1  elevation 1

4. PATH SCORE
   Casualty Scores = [5.856, 2.555, 2.072, 0.643, 0.468, 0.26, 0.149, 0.097, 0.074]
   TOTAL PATH SCORE = 12.175

6. TIME
   Total distance travelled = 6224.29 px
   TOTAL TIME = 359.21 seconds
```

---

## 4. Error analysis (the mistakes I made and how I fixed them)

1. **I forgot that the blue oval is a water body.** In my first version I only removed the
   black regions, so the rover was happily driving straight through the lake. Now I made a
   separate `find_water()` function and removed it from the traversable mask also.

2. **When I removed the water, my purple END triangle also disappeared.** Both are
   blue-ish so my first HSV range was taking both of them. I printed the HSV of both pixels
   and saw that the water has saturation 230 but the triangle has only 135, so I used the
   saturation to separate them.

3. **The three greens could not be separated by brightness.** I first tried simple
   brightness ranges, but when I printed `np.unique()` of the green pixels I got **hundreds**
   of different values instead of 3, because the edges of the ellipses are blurred
   (anti aliasing). So instead I find the **main flat green colours** (the ones covering more
   than 1% of the green area) and then give every green pixel to the nearest main colour.
   Now the blurred edge pixels also get put in the correct level.

4. **The order of the green levels was wrong.** I was sorting the shades by brightness but
   that gave the wrong level numbers. Then I noticed that the elevations are drawn like
   ellipses sitting inside each other, so the main ground always covers the **biggest area**
   and the higher levels cover smaller areas. Sorting by area works correctly.

5. **The stars were being detected as circles.** Just counting corners was not enough
   because `approxPolyDP` sometimes smooths the spikes. Adding the solidity check fixed it.

6. **A\* could not reach the casualties.** The shapes are not green, so they were making
   holes in my traversable mask and the casualty position was counted as an obstacle. I
   fixed it by adding all the shape masks back into the traversable area.

7. **`MORPH_CLOSE` was eating a bit of the thin black wall.** So after doing the closing I
   remove the black and water pixels one more time.


## 5. What can be improved
 Still Thinking!!
