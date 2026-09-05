# UAS-DTU Round 2 — Task 2 (Bonus Task)
## Walking to the target with a LIMITED RECEPTIVE FIELD

My solution file is **`task2.py`**. Only OpenCV, NumPy and normal python is used.


## 1. What the question is asking (in simple words)

I am given a maze picture :

| Colour | Meaning |
|---|---|
| Green | I can walk here |
| Red | Wall, I cannot pass |
| White circle | This is ME, my starting place |
| Black circle | The target I have to reach |

The catch is this line from the pdf :

> **ALL YOU HAVE ARE THE CO-ORDINATES OF THE STARTING AND ENDING POINT IN THE IMAGE
> WHICH YOU WILL NEED TO FIND YOURSELF USING CV CODE, AFTER WHICH YOU WILL HAVE NO
> ACCESS TO ANY MORE GLOBAL DATA.**

So I am allowed to look at the **full image only ONE time**, just to find the two circles.
After that I am blind. I can only see a small window around myself called the
**receptive field**, which is 320*360 as given in the pdf


The other rules :
* I move exactly **20 pixels in one step**
* Only **up, down, left, right**. If I want to go diagonally I have to take **two steps(40 pixel)**.
* I cannot pass through the red walls
* Save a **snapshot after every step**
* At the end join all the snapshots into a **video**

---

## 2. My idea (how a blind robot can still find the way)

I cannot see the whole maze, but nobody said I cannot **remember** what I have already
seen. So I keep a **memory map** which is the size of the full image and starts completely
empty (unknown). Every step :

1. I crop my receptive field and look at it
2. Whatever I saw, I paste into my memory map (green → FREE, red → WALL)
3. I plan a path on my **memory**, not on the real image
4. I take one step of 20 px, save a snapshot, and repeat

While planning, the parts I have **not seen yet are counted as FREE**. This means I am
hopeful that there is a way through them. If I walk there and find out it is actually a
wall, my memory gets corrected and I plan a new path. This is exactly how a real robot
explores an unknown place, and it is called optimistic planning.

```
memory :   0 = UNKNOWN (not seen yet, I hope it is free)
           1 = FREE
           2 = WALL
```

I also make the walls **fatter** by my own radius (`cv2.dilate`) so that the centre of my
circle never goes so close that my body touches the wall.

---

## 3. Flowchart

   UAS_Handwritten_stuff.pdf

---

## 4. Output

```
IMAGE : maze1  size = 1280 x 720
Receptive field size = 320 x 360
White circle (start) = (153, 629)
Black circle (target)= (1184, 149)
my radius = 38
--- from here I can only see my receptive field ---
REACHED the black circle in 118 steps
total steps walked = 115
```

Files made in `outputs/` :

| File | What it is |
|---|---|
| `maze1_steps/step_0000.png ...` | one snapshot per step (119 of them) |
| `maze1_video.mp4` | all the snapshots joined |
| `maze1_final_path.png` | the full route drawn on the maze |

In every snapshot, the part **inside the yellow box is the receptive field** (the only
thing the robot can see). Everything outside is made dark on purpose, to show that the
robot is blind there.

---

## 5. Error analysis (my mistakes and how I fixed them)

1. **The image has a lot of noise.** My first red and green masks came out full of small
   dots and holes, and the robot thought there were walls everywhere. I fixed it with
   `cv2.medianBlur()` before making the mask, and `MORPH_CLOSE` after it.

2. **The robot got stuck going up and down forever.** This was my biggest bug. I was
   planning a path but only using the **first move** of it, and then throwing the plan
   away and planning again from scratch. Near a corner the plan kept changing between
   "go up" and "go down", so the robot just oscillated between two points and ran out of
   steps at 400 without reaching. **Fix :** now I plan the **full path** and keep
   following it, and only plan again when my memory says the path ahead is blocked. After
   this the robot reached in 118 steps.

3. **The robot was scraping the walls.** The white circle has a radius of about 38 px, but
   I was planning as if it was a single point, so the body was going inside the red. I
   fixed it by making the walls fatter with `cv2.dilate()` by my own radius before saving
   them into the memory map.

4. **BFS on every pixel was very slow.** I plan on a grid of 10 px instead of 1 px
   (`grid = 10`). A cell counts as blocked if there is any wall pixel inside it, which is
   the safe way to do it.

5. **The black target and the black outline of my own circle were mixing up.** I take the
   **biggest** black contour only, so the small outline never wins.

## 6. What can be improved
* Still Thinking!!
