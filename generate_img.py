"""
Image generator for Sign Language Interpreter project
By Syeda Fizzah Batool
Generates 5 branded PNG assets for the img/ folder.
"""

from PIL import Image, ImageDraw, ImageFont
import math, os

OUT = "img"
os.makedirs(OUT, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = (10, 10, 30)        # near-black navy
CARD     = (20, 20, 50)        # dark card
P1       = (124, 58, 237)      # purple
P2       = (59, 130, 246)      # blue
CYAN     = (6,  182, 212)
WHITE    = (255, 255, 255)
LGRAY    = (200, 210, 230)
DGRAY    = (100, 120, 150)
GOLD     = (250, 200, 60)
GREEN    = (52,  211, 153)

# ── Fonts ────────────────────────────────────────────────────────────────────
def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

BOLD   = r"C:\Windows\Fonts\arialbd.ttf"
REG    = r"C:\Windows\Fonts\arial.ttf"
ITALIC = r"C:\Windows\Fonts\ariali.ttf"

# ── Helpers ──────────────────────────────────────────────────────────────────
def gradient_bg(draw, w, h, c1=BG, c2=(30, 10, 60)):
    for y in range(h):
        t = y / h
        r = int(c1[0] + t*(c2[0]-c1[0]))
        g = int(c1[1] + t*(c2[1]-c1[1]))
        b = int(c1[2] + t*(c2[2]-c1[2]))
        draw.line([(0,y),(w,y)], fill=(r,g,b))

def pill(draw, x, y, w, h, color, radius=14):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=color)

def centered(draw, text, y, fnt, color, W):
    bbox = draw.textbbox((0,0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    draw.text(((W-tw)//2, y), text, font=fnt, fill=color)

def dot_grid(draw, w, h, spacing=40, r=1):
    for x in range(0, w, spacing):
        for y in range(0, h, spacing):
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255,255,255,25))

def grad_pill_horizontal(img, x, y, w, h, c1, c2, radius=14):
    """Draw a horizontally-gradient pill."""
    strip = Image.new("RGBA", (w, h))
    sd = ImageDraw.Draw(strip)
    for i in range(w):
        t = i / w
        r = int(c1[0]*(1-t) + c2[0]*t)
        g = int(c1[1]*(1-t) + c2[1]*t)
        b = int(c1[2]*(1-t) + c2[2]*t)
        sd.line([(i,0),(i,h)], fill=(r,g,b,255))
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0,0,w,h], radius=radius, fill=255)
    strip.putalpha(mask)
    img.paste(strip, (x, y), strip)

# ═════════════════════════════════════════════════════════════════════════════
# 1. HERO BANNER
# ═════════════════════════════════════════════════════════════════════════════
W, H = 960, 480
img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)
gradient_bg(draw, W, H, (8,8,28), (40,10,80))

# decorative circles
for cx, cy, rad, alpha in [(750,80,180,18),(820,380,120,12),(120,400,100,10)]:
    ring = Image.new("RGBA",(W,H),(0,0,0,0))
    rd = ImageDraw.Draw(ring)
    for dr in range(4):
        rd.arc([cx-rad+dr, cy-rad+dr, cx+rad-dr, cy+rad-dr], 0, 360, fill=(*P1, alpha))
    img = img.convert("RGBA"); img = Image.alpha_composite(img, ring); img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

# dot grid subtle
dot_grid(draw, W, H, 50, 1)

# gradient accent bar top
grad_pill_horizontal(img.convert("RGBA"), 60, 48, 840, 6, P1, P2, 3)
img = img.convert("RGB"); draw = ImageDraw.Draw(img)

# emoji + title
centered(draw, "🤟  Sign Language Interpreter", 80, font(BOLD, 52), WHITE, W)

# subtitle gradient pill behind "Using Deep Learning"
grad_pill_horizontal(img.convert("RGBA"), W//2-200, 150, 400, 46, P1, P2, 23)
img = img.convert("RGB"); draw = ImageDraw.Draw(img)
centered(draw, "Using Deep Learning", 160, font(BOLD, 26), WHITE, W)

# author line
centered(draw, "By  Syeda Fizzah Batool", 222, font(BOLD, 30), CYAN, W)

# tagline
centered(draw, "Real-time ASL interpretation  ·  Convolutional Neural Network  ·  OpenCV", 276, font(REG, 18), LGRAY, W)

# stat pills
stats = [("44", "ASL Characters"), (">95%", "Accuracy"), ("1200+", "Images/Gesture"), ("Real-time", "Recognition")]
sw = 180; gap = 20; total = len(stats)*sw + (len(stats)-1)*gap
sx = (W-total)//2
for i,(val,lbl) in enumerate(stats):
    px = sx + i*(sw+gap)
    pill(draw, px, 330, sw, 80, CARD, 14)
    draw.rounded_rectangle([px, 330, px+sw, 332], radius=0, fill=P1)
    centered_in = lambda txt, y2, f, c: draw.text(
        (px + (sw - (draw.textbbox((0,0),txt,font=f)[2]-draw.textbbox((0,0),txt,font=f)[0]))//2, y2),
        txt, font=f, fill=c)
    centered_in(val, 345, font(BOLD, 24), GOLD)
    centered_in(lbl, 380, font(REG, 14), LGRAY)

# bottom bar
grad_pill_horizontal(img.convert("RGBA"), 60, 426, 840, 6, P2, P1, 3)
img = img.convert("RGB"); draw = ImageDraw.Draw(img)
centered(draw, "MIT License  ·  Python  ·  TensorFlow  ·  Keras  ·  OpenCV  ·  Streamlit", 440, font(REG, 15), DGRAY, W)

img.save(f"{OUT}/hero.png")
print("✅ hero.png")

# ═════════════════════════════════════════════════════════════════════════════
# 2. HOW IT WORKS  (pipeline)
# ═════════════════════════════════════════════════════════════════════════════
W, H = 980, 340
img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)
gradient_bg(draw, W, H, (8,8,28), (30,8,70))
dot_grid(draw, W, H, 45, 1)

centered(draw, "How It Works", 22, font(BOLD, 34), WHITE, W)
draw.line([(W//2-140, 64),(W//2+140, 64)], fill=P1, width=2)

steps = [
    ("📷","Webcam\nInput",    P1),
    ("🖐","Hand\nDetection",  P2),
    ("🧠","CNN\nModel",       (168,85,247)),
    ("🔤","Gesture\nPrediction", CYAN),
    ("🔊","Speech\nOutput",   GREEN),
]

bw, bh = 140, 140
gap2 = 20
total2 = len(steps)*bw + (len(steps)-1)*gap2
sx2 = (W-total2)//2

for i,(ico,lbl,col) in enumerate(steps):
    px = sx2 + i*(bw+gap2)
    py = 90
    # box
    pill(draw, px, py, bw, bh, CARD, 16)
    draw.rounded_rectangle([px, py, px+bw, py+4], radius=2, fill=col)
    # icon
    centered(draw, ico, py+14, font(REG, 36), WHITE, px + bw//2*0 + W//2*0)
    bbox = draw.textbbox((0,0), ico, font=font(REG,36))
    iw = bbox[2]-bbox[0]
    draw.text((px+(bw-iw)//2, py+12), ico, font=font(REG,36), fill=WHITE)
    # label
    for j, line in enumerate(lbl.split("\n")):
        bbox2 = draw.textbbox((0,0), line, font=font(BOLD,15))
        lw = bbox2[2]-bbox2[0]
        draw.text((px+(bw-lw)//2, py+62+j*20), line, font=font(BOLD,15), fill=WHITE)
    # step number
    pill(draw, px+bw-26, py-10, 24, 24, col, 12)
    draw.text((px+bw-19, py-8), str(i+1), font=font(BOLD,13), fill=WHITE)
    # arrow
    if i < len(steps)-1:
        ax = px+bw+4
        ay = py+bh//2
        draw.line([(ax,ay),(ax+gap2-4,ay)], fill=LGRAY, width=2)
        draw.polygon([(ax+gap2-4,ay-5),(ax+gap2-4,ay+5),(ax+gap2+4,ay)], fill=LGRAY)

centered(draw, "Confidence threshold > 70%  ·  20 consecutive frames required to confirm gesture", 248, font(REG, 15), DGRAY, W)
centered(draw, "By Syeda Fizzah Batool", 278, font(ITALIC, 14), P1, W)

img.save(f"{OUT}/pipeline.png")
print("✅ pipeline.png")

# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURES GRID
# ═════════════════════════════════════════════════════════════════════════════
W, H = 900, 480
img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)
gradient_bg(draw, W, H, (8,8,28), (25,5,55))
dot_grid(draw, W, H, 45, 1)

centered(draw, "Key Features", 22, font(BOLD, 34), WHITE, W)
draw.line([(W//2-120,62),(W//2+120,62)], fill=CYAN, width=2)

feats = [
    ("📷","Real-time Recognition",  "Processes live webcam feed at\nfull frame rate with OpenCV",            P1),
    ("✍️","Text Mode",              "Build words and sentences from\nASL gestures, letter by letter",        P2),
    ("🔢","Calculator Mode",        "Perform arithmetic using hand\ngestures for digits and operators",      (168,85,247)),
    ("🔊","Text-to-Speech",         "Spoken output via pyttsx3\nToggle voice on/off with 'V' key",           CYAN),
    ("🧠","CNN Architecture",       "3 Conv layers → Dense 128 → Softmax\n>95% accuracy on 44 characters",  GREEN),
    ("🛡","Confidence Filter",      "Only accepts predictions above\n70% probability threshold",             GOLD),
]

cols2 = 3; rows2 = 2
fw = 260; fh = 155; fgx = 20; fgy = 18
total_w = cols2*fw + (cols2-1)*fgx
sx3 = (W-total_w)//2
sy3 = 80

for i,(ico,title,desc,col) in enumerate(feats):
    r, c = divmod(i, cols2)
    px = sx3 + c*(fw+fgx)
    py = sy3 + r*(fh+fgy)
    pill(draw, px, py, fw, fh, CARD, 14)
    draw.rounded_rectangle([px, py, px+fw, py+4], radius=2, fill=col)
    # icon pill
    pill(draw, px+14, py+16, 44, 44, (*col[:3],), 10)
    bbox = draw.textbbox((0,0), ico, font=font(REG,24))
    iw = bbox[2]-bbox[0]
    draw.text((px+14+(44-iw)//2, py+18), ico, font=font(REG,24), fill=WHITE)
    # title
    draw.text((px+66, py+18), title, font=font(BOLD,16), fill=WHITE)
    # desc
    for j, ln in enumerate(desc.split("\n")):
        draw.text((px+14, py+70+j*20), ln, font=font(REG,13), fill=LGRAY)

centered(draw, "By Syeda Fizzah Batool  ·  MIT License", 462, font(ITALIC,14), DGRAY, W)
img.save(f"{OUT}/features.png")
print("✅ features.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4. CNN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
W, H = 940, 380
img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)
gradient_bg(draw, W, H, (8,8,28), (30,8,60))
dot_grid(draw, W, H, 45, 1)

centered(draw, "CNN Model Architecture", 20, font(BOLD,32), WHITE, W)
draw.line([(W//2-160,58),(W//2+160,58)], fill=P2, width=2)

layers = [
    ("Input\n50×50×1",       (40,40,40),   60,  200),
    ("Conv2D 16\n+ MaxPool",  P1,           100, 180),
    ("Conv2D 32\n+ MaxPool",  (100,60,200), 120, 160),
    ("Conv2D 64\n+ MaxPool",  P2,           140, 140),
    ("Flatten",               (168,85,247), 80,  80),
    ("Dense 128\nReLU",       CYAN,         90,  90),
    ("Dropout\n0.2",          (200,80,80),  80,  80),
    ("Softmax\nN classes",    GREEN,        90,  90),
]

lw_each = 96; gap3 = 10
total3 = len(layers)*lw_each + (len(layers)-1)*gap3
sx4 = (W-total3)//2
base_y = H//2

for i,(lbl,col,lh,lwd) in enumerate(layers):
    px = sx4 + i*(lw_each+gap3)
    py = base_y - lh//2
    # 3-D block effect
    draw.rectangle([px+6, py+6, px+lwd-6+6, py+lh+6], fill=(0,0,0,0))
    draw.rectangle([px+4, py+4, px+lwd-6+4, py+lh+4], fill=tuple(max(0,c-60) for c in col))
    pill(draw, px, py, lwd-6, lh, col, 10)
    for j, ln in enumerate(lbl.split("\n")):
        bbox = draw.textbbox((0,0), ln, font=font(BOLD,12))
        lw2 = bbox[2]-bbox[0]
        draw.text((px+(lwd-6-lw2)//2, py+lh//2-14+j*18), ln, font=font(BOLD,12), fill=WHITE)
    # arrow
    if i < len(layers)-1:
        ax = px+lwd-6
        ay = base_y
        draw.line([(ax,ay),(ax+gap3,ay)], fill=LGRAY, width=2)
        draw.polygon([(ax+gap3-2,ay-4),(ax+gap3-2,ay+4),(ax+gap3+4,ay)], fill=LGRAY)

centered(draw, "Optimizer: SGD  ·  Loss: Categorical Crossentropy  ·  Epochs: 15  ·  Batch: 500", 320, font(REG,15), LGRAY, W)
centered(draw, "By Syeda Fizzah Batool", 345, font(ITALIC,14), P1, W)
img.save(f"{OUT}/architecture.png")
print("✅ architecture.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. TECH STACK
# ═════════════════════════════════════════════════════════════════════════════
W, H = 900, 320
img = Image.new("RGB", (W, H))
draw = ImageDraw.Draw(img)
gradient_bg(draw, W, H, (8,8,28), (20,5,50))
dot_grid(draw, W, H, 45, 1)

centered(draw, "Technologies & Tools", 20, font(BOLD,32), WHITE, W)
draw.line([(W//2-150,58),(W//2+150,58)], fill=GOLD, width=2)

techs = [
    ("🐍","Python 3",        (55,118,171)),
    ("🔷","TensorFlow",      (255,144,0)),
    ("🔶","Keras",            (210,60,60)),
    ("👁","OpenCV",           (50,200,100)),
    ("📊","SQLite",           (100,160,220)),
    ("🌐","Streamlit",        (255,75,75)),
    ("🔊","pyttsx3",          (168,85,247)),
    ("🔢","NumPy",            (77,171,207)),
]

tw2 = 96; th2 = 120; tgx2 = 10
total4 = len(techs)*tw2 + (len(techs)-1)*tgx2
sx5 = (W-total4)//2
ty = 88

for i,(ico,name,col) in enumerate(techs):
    px = sx5 + i*(tw2+tgx2)
    pill(draw, px, ty, tw2, th2, CARD, 14)
    draw.rounded_rectangle([px, ty+th2-4, px+tw2, ty+th2], radius=2, fill=col)
    bbox = draw.textbbox((0,0), ico, font=font(REG,30))
    iw = bbox[2]-bbox[0]
    draw.text((px+(tw2-iw)//2, ty+14), ico, font=font(REG,30), fill=WHITE)
    bbox2 = draw.textbbox((0,0), name, font=font(BOLD,12))
    nw = bbox2[2]-bbox2[0]
    draw.text((px+(tw2-nw)//2, ty+60), name, font=font(BOLD,12), fill=WHITE)

centered(draw, "By Syeda Fizzah Batool  ·  Sign Language Interpreter using Deep Learning  ·  MIT License", 230, font(REG,14), DGRAY, W)

# big accent pill at bottom
grad_pill_horizontal(img.convert("RGBA"), 60, 268, 780, 32, P1, P2, 16)
img = img.convert("RGB"); draw = ImageDraw.Draw(img)
centered(draw, "Real-time  •  Accessible  •  Open Source", 275, font(BOLD,16), WHITE, W)

img.save(f"{OUT}/tech_stack.png")
print("✅ tech_stack.png")
print("\nAll 5 images generated in img/")
