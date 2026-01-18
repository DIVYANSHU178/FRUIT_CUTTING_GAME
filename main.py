import sys
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pygame
import numpy as np
import random
import math
import json
import threading
import time
from pygame import mixer

# -------- SAFETY: AVOID LOCAL NAME CONFLICTS WITH MEDIAPIPE ----------
if 'mediapipe.py' in os.listdir(os.getcwd()):
    print("ERROR: Remove or rename local 'mediapipe.py' in this folder.")
    sys.exit(1)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------------- PYGAME / WINDOW SETUP -----------------
pygame.init()
mixer.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Fruit Ninja - HD")
clock = pygame.time.Clock()
font = pygame.font.Font("font/slkscreb.ttf", 40)
big_font = pygame.font.Font("font/slkscreb.ttf", 80)

# ------------- COLORS -------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW= (255, 255, 0)
CYAN  = (0, 255, 255)

# ------------- IMAGE LOADER -------------
def load_image(path, size=None):
    try:
        img = pygame.image.load(path).convert_alpha()
        if size is not None:
            img = pygame.transform.smoothscale(img, size)
        return img
    except pygame.error as e:
        print(f"Warning: Could not load image {path}: {e}")
        return None

# ---------- ASSET SIZES ----------
TARGET_FRUIT_SIZE = (120, 120)
SPLASH_SIZE       = (150, 150)
BOMB_SIZE         = (100, 100)

# ---------- BACKGROUND ----------
background = load_image("assets/background.png.jpg", (WIDTH, HEIGHT))

# ---------- FRUIT ASSETS ----------
assets = {}

def load_fruit_data(name, points, splash_name):
    base = f"assets/fruits/{name}"
    data = {
        "points": points,
        "img":    load_image(f"{base}.png",           TARGET_FRUIT_SIZE),
        "halves": [
            load_image(f"{base}_half_1.png", TARGET_FRUIT_SIZE),
            load_image(f"{base}_half_2.png", TARGET_FRUIT_SIZE),
        ],
        "splash": load_image(f"assets/splashes/{splash_name}.png", SPLASH_SIZE),
        "is_bomb": False
    }
    return data

assets["apple"]      = load_fruit_data("apple",      10, "splash_red")
assets["banana"]     = load_fruit_data("banana",     15, "splash_yellow")
assets["orange"]     = load_fruit_data("orange",     12, "splash_orange")
assets["watermelon"] = load_fruit_data("watermelon", 20, "splash_green")

assets["bomb"] = {
    "points": -10,
    "img": load_image("assets/fruits/bomb.png", BOMB_SIZE),
    "explosion": load_image("assets/splashes/explosion.png", SPLASH_SIZE),
    "is_bomb": True
}
if assets["bomb"]["img"] is None:
    bomb_fallback = pygame.Surface(BOMB_SIZE, pygame.SRCALPHA)
    pygame.draw.circle(bomb_fallback, (0, 0, 0), (BOMB_SIZE[0]//2, BOMB_SIZE[1]//2), BOMB_SIZE[0]//2)
    assets["bomb"]["img"] = bomb_fallback

# ------------- SOUNDS -------------
slice_sound = bomb_sound = None
try:
    mixer.music.load("sounds/bgm.mp3")
    mixer.music.set_volume(0.4)
    mixer.music.play(-1)
    slice_sound = mixer.Sound("sounds/slice.wav")
    bomb_sound  = mixer.Sound("sounds/bomb.wav")
except:
    pass

# ------------- GAME STATE -------------
MENU, PLAYING, GAME_OVER, PAUSED = 0, 1, 2, 3
state = MENU
timer_options = [60, 120]
selected_timer_index = 0
time_left = timer_options[selected_timer_index]

score = 0
high_score = 0
lives = 3
combo = 1
last_slice_time = 0.0

if os.path.exists("highscore.json"):
    try:
        with open("highscore.json", "r") as f:
            high_score = json.load(f).get("score", 0)
    except:
        high_score = 0

# ------------- MEDIAPIPE HANDS (FROM SECOND CODE STYLE) -------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

# ------------- CAMERA -------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

# Borders for camera view scaling
camera_border_x = 0.2
camera_border_y = 0.2


# ------------- CLASSES FROM FIRST CODE -------------
class Splash:
    def __init__(self, x, y, img, duration=0.5):
        self.x = x
        self.y = y
        self.img = img
        self.birth = time.time()
        self.duration = duration

    def alive(self):
        return time.time() - self.birth < self.duration

    def draw(self, surface):
        if not self.img:
            return
        alpha = 255 * (1 - (time.time() - self.birth) / self.duration)
        img_to_draw = self.img.copy()
        img_to_draw.set_alpha(max(0, int(alpha)))
        rect = img_to_draw.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(img_to_draw, rect)

class Fruit:
    def __init__(self, ftype):
        self.ftype = ftype
        self.data = assets[ftype]
        self.img  = self.data["img"]
        self.rect = self.img.get_rect() if self.img else pygame.Rect(0,0,*TARGET_FRUIT_SIZE)

        self.x = random.randint(100, WIDTH - 100)
        self.y = HEIGHT + self.rect.height // 2
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-30, -22) if ftype != "watermelon" else random.uniform(-25, -18)
        self.angle = 0
        self.angular_speed = random.uniform(-1.5, 1.5)

        self.alive = True
        self.sliced = False
        self.half_vx = 0
        self.half_vy = 0
        self.sliced_time = 0.0

    def update(self, dt):
        if not self.alive:
            return

        if not self.sliced:
            self.x += self.vx * dt * 60
            self.y += self.vy * dt * 60
            self.vy += 0.5 * dt * 60
            self.angle += self.angular_speed * dt * 60
        else:
            self.x += self.half_vx * dt * 60
            self.y += self.half_vy * dt * 60
            self.half_vy += 0.7 * dt * 60
            self.angle += self.angular_speed * dt * 60

        if self.y > HEIGHT + self.rect.height:
            self.alive = False

    def draw(self, surface):
        if not self.alive:
            return

        if not self.sliced:
            if self.img:
                rotated = pygame.transform.rotozoom(self.img, self.angle, 1.0)
                rect = rotated.get_rect(center=(int(self.x), int(self.y)))
                surface.blit(rotated, rect)
        else:
            half1 = self.data["halves"][0]
            half2 = self.data["halves"][1]
            if half1 and half2:
                half1_img = pygame.transform.rotozoom(half1, self.angle + 15, 1.0)
                half2_img = pygame.transform.rotozoom(half2, self.angle - 15, 1.0)
                offset_x = self.rect.width // 4
                offset_y = self.rect.height // 4
                rect1 = half1_img.get_rect(center=(int(self.x - offset_x), int(self.y - offset_y)))
                rect2 = half2_img.get_rect(center=(int(self.x + offset_x), int(self.y + offset_y)))
                surface.blit(half1_img, rect1)
                surface.blit(half2_img, rect2)

    def get_points(self):
        radius = self.rect.width // 2
        return (self.x, self.y, radius)

class Bomb:
    def __init__(self):
        self.data = assets["bomb"]
        self.img  = self.data["img"]
        self.rect = self.img.get_rect() if self.img else pygame.Rect(0,0,*BOMB_SIZE)

        self.x = random.randint(100, WIDTH - 100)
        self.y = HEIGHT + self.rect.height // 2
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-25, -18)
        self.angle = 0
        self.angular_speed = random.uniform(-1.5, 1.5)
        self.alive = True

    def update(self, dt):
        if not self.alive:
            return
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60
        self.vy += 0.5 * dt * 60
        self.angle += self.angular_speed * dt * 60
        if self.y > HEIGHT + self.rect.height:
            self.alive = False

    def draw(self, surface):
        if not self.alive:
            return
        rotated = pygame.transform.rotozoom(self.img, self.angle, 1.0)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)

    def get_points(self):
        radius = self.rect.width // 2
        return (self.x, self.y, radius)

# ------------- GLOBAL LISTS -------------
fruits = []
bombs = []
splashes = []

# ------------- SPAWN (FROM FIRST CODE) -------------
def spawn_entities(dt):
    if not hasattr(spawn_entities, "fruit_timer"):
        spawn_entities.fruit_timer = 0.0
        spawn_entities.bomb_timer  = 0.0

    spawn_entities.fruit_timer += dt
    spawn_entities.bomb_timer  += dt

    base_interval = 1.8
    difficulty_factor = max(0.6, 1.0 - score / 2000.0)
    fruit_interval = base_interval * difficulty_factor

    if spawn_entities.fruit_timer >= fruit_interval:
        spawn_entities.fruit_timer = 0.0
        if len(fruits) < 4:
            ftype = random.choice([f for f in assets.keys() if not assets[f]["is_bomb"]])
            fruits.append(Fruit(ftype))

    bomb_interval_factor = random.uniform(4.0, 6.0)
    bomb_interval = fruit_interval * bomb_interval_factor
    if spawn_entities.bomb_timer >= bomb_interval:
        spawn_entities.bomb_timer = 0.0
        if len(bombs) < 1:
            bombs.append(Bomb())

# ------------- GESTURE / CURSOR STATE (FROM SECOND CODE IDEA) -------------
cursor_pos = None
last_cursor_pos = None
position_history = []  # (x, y, t)
slashes = []
max_slash_points = 8
smoothing_factor = 0.2
min_movement_threshold = 3
max_smoothing_distance = 40
velocity_threshold = 80
fist_threshold = 0.15

def is_fist(hand_landmarks):
    # Heuristic to check for a fist
    # Compare distance of fingertips to the wrist
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    
    # Calculate distance
    dist = math.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2 + (index_tip.z - wrist.z)**2)
    
    return dist < fist_threshold

def smooth_position(x, y):
    global last_cursor_pos
    if last_cursor_pos is None:
        last_cursor_pos = (x, y)
        return x, y

    lx, ly = last_cursor_pos
    dx, dy = x - lx, y - ly
    distance = (dx*dx + dy*dy) ** 0.5

    if distance < min_movement_threshold:
        return lx, ly

    if distance > max_smoothing_distance:
        adaptive = min(0.8, smoothing_factor * 2)
    else:
        adaptive = smoothing_factor

    sx = adaptive * x + (1 - adaptive) * lx
    sy = adaptive * y + (1 - adaptive) * ly
    last_cursor_pos = (sx, sy)
    return int(sx), int(sy)

def process_gestures():
    """
    Use MediaPipe Hands (solutions) like second code:
    - Track index fingertip
    - Smooth its position
    - Build a velocity-based slash trail
    """
    global cursor_pos, position_history, slashes

    ret, frame = cap.read()
    if not ret:
        cursor_pos = None
        return

    frame = cv2.flip(frame, 1)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = landmarker.detect(image)

    if not detection_result.hand_landmarks:
        cursor_pos = None
        slashes.clear()
        return

    hand_landmarks = detection_result.hand_landmarks[0]
    
    if is_fist(hand_landmarks):
        slashes.clear()
        return # Stop processing if fist is detected

    h, w, _ = frame.shape
    wrist = hand_landmarks[0]
    
    # Scale the coordinates
    raw_x = np.interp(wrist.x, [camera_border_x, 1 - camera_border_x], [0, WIDTH])
    raw_y = np.interp(wrist.y, [camera_border_y, 1 - camera_border_y], [0, HEIGHT])


    x, y = smooth_position(raw_x, raw_y)
    cursor_pos = (x, y)

    now = time.time()
    position_history.append((x, y, now))
    if len(position_history) > 8:
        position_history.pop(0)

    if len(position_history) >= 2:
        px, py, pt = position_history[-2]
        dt = now - pt
        if dt > 0:
            v = ((x - px)**2 + (y - py)**2) ** 0.5 / dt
            if v > velocity_threshold:
                slashes.append((x, y))
                if len(slashes) > max_slash_points:
                    slashes.pop(0)
            elif len(slashes) > 0:
                slashes.pop(0)


def camera_loop():
    while True:
        if state == PLAYING:
            process_gestures()
        time.sleep(0.001)

threading.Thread(target=camera_loop, daemon=True).start()

# ------------- COLLISION & SLICES (MERGED) -------------
def point_circle_collide(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2

def handle_slices():
    global combo, score, last_slice_time, lives

    if not slashes:
        return

    now = time.time()
    slashed_this_frame = False

    # Fruits
    for f in fruits[:]:
        if not f.alive or f.sliced:
            continue
        fx, fy, fr = f.get_points()
        hit = any(point_circle_collide(px, py, fx, fy, fr + 8) for (px, py) in slashes)
        if hit:
            slashed_this_frame = True
            base_points = f.data["points"]
            if now - last_slice_time < 0.3:
                combo = min(combo + 1, 5)
            else:
                combo = 1
            last_slice_time = now
            score += base_points * combo

            if slice_sound:
                slice_sound.play()

            if f.data["splash"]:
                splashes.append(Splash(f.x, f.y, f.data["splash"], duration=0.4))

            f.sliced = True
            f.sliced_time = now
            f.img = None
            f.half_vx = random.uniform(-4, 4)
            f.half_vy = random.uniform(-8, -4)

    # Bombs
    for b in bombs[:]:
        bx, by, br = b.get_points()
        hit = any(point_circle_collide(px, py, bx, by, br + 8) for (px, py) in slashes)
        if hit:
            slashed_this_frame = True
            lives -= 1
            combo = 1
            if bomb_sound:
                bomb_sound.play()
            bombs.remove(b)
            splashes.append(Splash(b.x, b.y, assets["bomb"]["explosion"], duration=0.8))
    
    if slashed_this_frame:
        slashes.clear()


# ------------- DRAWING -------------
def draw_background():
    if background:
        screen.blit(background, (0, 0))
    else:
        screen.fill(BLACK)

def draw_blade_and_pointer():
    if len(slashes) > 1:
        for i in range(1, len(slashes)):
            p1 = slashes[i - 1]
            p2 = slashes[i]
            alpha = i / len(slashes)
            color = (0, int(255 * alpha), 255)
            width = int(14 * alpha)
            if width > 0:
                pygame.draw.line(screen, color, p1, p2, width)
    if cursor_pos:
        pygame.draw.circle(screen, CYAN, cursor_pos, 12, 2)

def draw_hud():
    score_text = font.render(f"Score: {score}  Combo x{combo}", True, WHITE)
    time_text  = font.render(f"Time: {int(time_left)}s", True, WHITE)
    lives_text = font.render(f"Lives: {lives}", True, RED)
    screen.blit(score_text, (20, 20))
    screen.blit(time_text,  (20, 60))
    screen.blit(lives_text, (20, 100))
    pause_text = font.render("P: Pause", True, YELLOW)
    screen.blit(pause_text, (WIDTH - pause_text.get_width() - 20, 20))

# ------------- MAIN LOOP -------------
running = True
last_frame_time = time.time()
key_cooldown = 0

while running:
    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now
    key_cooldown = max(0, key_cooldown - dt)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if state == MENU:
                if event.key == pygame.K_SPACE:
                    state = PLAYING
                    time_left = timer_options[selected_timer_index]
                    score = 0
                    lives = 3
                    combo = 1
                    fruits.clear()
                    bombs.clear()
                    splashes.clear()
            elif state == GAME_OVER:
                if event.key == pygame.K_RETURN:
                    state = PLAYING
                    time_left = timer_options[selected_timer_index]
                    score = 0
                    lives = 3
                    combo = 1
                    fruits.clear()
                    bombs.clear()
                    splashes.clear()
                elif event.key == pygame.K_q:
                    running = False
            elif state == PLAYING:
                if event.key == pygame.K_p:
                    state = PAUSED
            elif state == PAUSED:
                if event.key == pygame.K_p:
                    state = PLAYING

    keys = pygame.key.get_pressed()
    if state == MENU and key_cooldown == 0:
        if keys[pygame.K_z]:
            selected_timer_index = (selected_timer_index - 1) % len(timer_options)
            key_cooldown = 0.25
        if keys[pygame.K_c]:
            selected_timer_index = (selected_timer_index + 1) % len(timer_options)
            key_cooldown = 0.25

    draw_background()

    if state == MENU:
        title = big_font.render("Fruit Ninja", True, WHITE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3 - 60))

        info = font.render("Use your hand to slice fruits via webcam!", True, WHITE)
        screen.blit(info, (WIDTH//2 - info.get_width()//2, HEIGHT//3 + 20))

        timer_text = font.render(
            f"Timer: {timer_options[selected_timer_index]}s  (Z/C to change)",
            True,
            YELLOW
        )
        screen.blit(timer_text, (WIDTH//2 - timer_text.get_width()//2, HEIGHT//2))

        start_text = font.render("Press SPACE to Start", True, GREEN)
        screen.blit(start_text, (WIDTH//2 - start_text.get_width()//2, HEIGHT//2 + 60))

        hs_text = font.render(f"High Score: {high_score}", True, (0, 200, 255))
        screen.blit(hs_text, (WIDTH//2 - hs_text.get_width()//2, HEIGHT//2 + 120))

    elif state == PLAYING:
        time_left -= dt
        if time_left <= 0 or lives <= 0:
            state = GAME_OVER

        spawn_entities(dt)

        for f in fruits[:]:
            f.update(dt)
            if f.sliced and now - f.sliced_time > 1.0:
                fruits.remove(f)
        fruits[:] = [f for f in fruits if f.alive]

        for b in bombs:
            b.update(dt)
        bombs[:] = [b for b in bombs if b.alive]

        handle_slices()

        for s in splashes[:]:
            if not s.alive():
                splashes.remove(s)
            else:
                s.draw(screen)

        for f in fruits:
            f.draw(screen)
        for b in bombs:
            b.draw(screen)

        draw_blade_and_pointer()
        draw_hud()

    elif state == PAUSED:
        paused_text = big_font.render("PAUSED", True, WHITE)
        screen.blit(paused_text, (WIDTH//2 - paused_text.get_width()//2, HEIGHT//2 - 40))
        resume_text = font.render("Press P to Resume", True, GREEN)
        screen.blit(resume_text, (WIDTH//2 - resume_text.get_width()//2, HEIGHT//2 + 40))

    elif state == GAME_OVER:
        if score > high_score:
            high_score = score
            try:
                with open("highscore.json", "w") as f:
                    json.dump({"score": high_score}, f)
            except:
                pass

        over_text = big_font.render("GAME OVER", True, RED)
        screen.blit(over_text, (WIDTH//2 - over_text.get_width()//2, HEIGHT//3 - 40))

        s_text = font.render(f"Final Score: {score}", True, WHITE)
        screen.blit(s_text, (WIDTH//2 - s_text.get_width()//2, HEIGHT//3 + 40))

        hs_text = font.render(f"High Score: {high_score}", True, YELLOW)
        screen.blit(hs_text, (WIDTH//2 - hs_text.get_width()//2, HEIGHT//3 + 90))

        hint_text = font.render("Press Enter to Replay or Q to Quit", True, GREEN)
        screen.blit(hint_text, (WIDTH//2 - hint_text.get_width()//2, HEIGHT//3 + 160))

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
