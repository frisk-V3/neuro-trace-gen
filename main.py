#!/usr/bin/env python3
"""
main.py
画像トレース -> brain_data.txt 保存 -> brainから自律生成
単一ファイルで動作するサンプル実装
"""

import os
import json
import time
import math
import random
from datetime import datetime
from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from noise import pnoise2

# ---------------------------
# 設定
# ---------------------------
BRAIN_FILE = "brain_data.txt"
GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)

# ---------------------------
# ユーティリティ
# ---------------------------
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def save_brain_entry(entry: dict):
    with open(BRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_brain_entries():
    if not os.path.exists(BRAIN_FILE):
        return []
    with open(BRAIN_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    entries = []
    for L in lines:
        try:
            entries.append(json.loads(L))
        except Exception:
            continue
    return entries

# ---------------------------
# Tracer: 画像から特徴抽出
# ---------------------------
class Tracer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path).convert("RGB")

    def _to_np(self, img=None):
        if img is None:
            img = self.img
        return np.asarray(img).astype(np.float32) / 255.0

    def average_color(self):
        arr = self._to_np()
        avg = arr.reshape(-1, 3).mean(axis=0)
        return (avg * 255).astype(int).tolist()

    def color_variance(self):
        arr = self._to_np()
        var = arr.reshape(-1, 3).var(axis=0)
        return var.tolist()

    def brightness_stats(self):
        arr = self._to_np()
        # luminance approximation
        lum = (0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2])
        return {
            "mean": float(lum.mean()),
            "var": float(lum.var()),
            "min": float(lum.min()),
            "max": float(lum.max())
        }

    def edge_density(self):
        # simple edge detection via PIL filter -> count strong gradients
        gray = ImageOps.grayscale(self.img).resize((256,256))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges).astype(np.float32) / 255.0
        density = float((arr > 0.2).mean())
        return density

    def dominant_colors(self, n=3, sample=10000):
        arr = (self._to_np() * 255).astype(np.uint8).reshape(-1,3)
        if arr.shape[0] > sample:
            idx = np.random.choice(arr.shape[0], sample, replace=False)
            arr = arr[idx]
        # simple histogram by packing RGB into int
        packed = (arr[:,0].astype(np.int32) << 16) + (arr[:,1].astype(np.int32) << 8) + arr[:,2].astype(np.int32)
        vals, counts = np.unique(packed, return_counts=True)
        order = np.argsort(-counts)
        top = vals[order][:n]
        colors = []
        for v in top:
            r = (v >> 16) & 255
            g = (v >> 8) & 255
            b = v & 255
            colors.append([int(r), int(g), int(b)])
        return colors

    def mood_vector(self):
        # 簡易的な「雰囲気」数値化: warmth, contrast, complexity
        avg = np.array(self.average_color()) / 255.0
        brightness = self.brightness_stats()
        warmth = float(avg[0] - avg[2])  # 赤 - 青
        contrast = float(brightness["max"] - brightness["min"])
        complexity = float(self.edge_density())
        # 正規化（簡易）
        return {
            "warmth": warmth,
            "contrast": contrast,
            "complexity": complexity
        }

    def extract_all(self):
        return {
            "source": os.path.basename(self.image_path),
            "timestamp": now_ts(),
            "avg_color": self.average_color(),
            "color_variance": self.color_variance(),
            "brightness": self.brightness_stats(),
            "edge_density": self.edge_density(),
            "dominant_colors": self.dominant_colors(n=3),
            "mood": self.mood_vector()
        }

# ---------------------------
# Generator: brainから読み出して生成
# ---------------------------
class Generator:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    def _aggregate_brain(self, entries):
        if not entries:
            # デフォルトの中立的な設定
            return {
                "avg_color": [128,128,128],
                "dominant_colors": [[128,128,128]],
                "mood": {"warmth": 0.0, "contrast": 0.5, "complexity": 0.2}
            }
        # 平均化
        avg_colors = np.array([e["avg_color"] for e in entries], dtype=float)
        avg_color = np.mean(avg_colors, axis=0).astype(int).tolist()
        # dominant palette: flatten and pick most frequent by simple counting
        palette = []
        for e in entries:
            for c in e.get("dominant_colors", []):
                palette.append(tuple(c))
        # count
        from collections import Counter
        cnt = Counter(palette)
        most = [list(c) for c,_ in cnt.most_common(5)]
        # mood average
        moods = np.array([[e.get("mood",{}).get("warmth",0.0),
                           e.get("mood",{}).get("contrast",0.0),
                           e.get("mood",{}).get("complexity",0.0)] for e in entries], dtype=float)
        mood_avg = moods.mean(axis=0).tolist()
        return {
            "avg_color": avg_color,
            "palette": most if most else [avg_color],
            "mood": {"warmth": float(mood_avg[0]), "contrast": float(mood_avg[1]), "complexity": float(mood_avg[2])}
        }

    def _choose_params(self, agg):
        # シードは過去のmoodと時間から決定（再現性あり）
        base_seed = int((agg["mood"]["warmth"]*1000 + agg["mood"]["contrast"]*100 + agg["mood"]["complexity"]*10000)) & 0xffffffff
        seed = (base_seed ^ int(time.time())) & 0xffffffff
        # palette: shuffle and maybe blend with avg_color
        palette = agg.get("palette", [agg["avg_color"]])
        # convert to normalized floats
        palette_f = [[c/255.0 for c in p] for p in palette]
        return {"seed": seed, "palette": palette_f, "mood": agg["mood"]}

    def _perlin_map(self, seed, scale=6.0, octaves=4, persistence=0.5, lacunarity=2.0):
        # 2D Perlin noise map
        w, h = self.width, self.height
        arr = np.zeros((h, w), dtype=np.float32)
        random.seed(seed)
        base = random.randint(0, 10000)
        for y in range(h):
            for x in range(w):
                nx = x / w * scale
                ny = y / h * scale
                val = pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=base)
                arr[y, x] = val
        # normalize to 0..1
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        return arr

    def _map_to_palette(self, noise_map, palette):
        h, w = noise_map.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        n_levels = len(palette)
        for i, color in enumerate(palette):
            # color is normalized float triple
            pass
        # map by thresholds
        thresholds = np.linspace(0.0, 1.0, n_levels+1)
        for i in range(n_levels):
            mask = (noise_map >= thresholds[i]) & (noise_map < thresholds[i+1])
            col = np.array(palette[i]) * 255.0
            img[mask] = col.astype(np.uint8)
        return Image.fromarray(img)

    def generate(self, out_name=None, width=None, height=None):
        if width: self.width = width
        if height: self.height = height
        entries = load_brain_entries()
        agg = self._aggregate_brain(entries)
        params = self._choose_params(agg)

        # adjust perlin parameters based on mood
        mood = params["mood"]
        complexity = max(0.1, min(1.0, mood.get("complexity", 0.2)))
        contrast = max(0.1, min(1.0, mood.get("contrast", 0.5)))
        warmth = mood.get("warmth", 0.0)

        scale = 3.0 + complexity * 10.0
        octaves = int(1 + complexity * 5)
        persistence = 0.4 + (1.0 - complexity) * 0.3
        lacunarity = 1.8 + complexity * 1.5

        noise_map = self._perlin_map(seed=params["seed"], scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

        # build palette: bias palette colors by warmth and contrast
        palette = params["palette"]
        # if palette shorter than 3, expand by blending with avg_color
        if len(palette) < 3:
            avg = np.array(agg["avg_color"]) / 255.0
            while len(palette) < 3:
                palette.append(avg.tolist())
        # apply warmth bias
        def bias_color(c):
            c = np.array(c)
            c[0] = np.clip(c[0] + warmth*0.1, 0.0, 1.0)
            # contrast stretch
            c = (c - 0.5) * (1.0 + contrast) + 0.5
            return np.clip(c, 0.0, 1.0).tolist()
        palette = [bias_color(c) for c in palette[:5]]

        img = self._map_to_palette(noise_map, palette)

        # post-process: slight blur or sharpen based on complexity
        if complexity < 0.3:
            img = img.filter(ImageFilter.SMOOTH)
        elif complexity > 0.7:
            img = img.filter(ImageFilter.DETAIL)

        if out_name is None:
            out_name = f"gen_{int(time.time())}.png"
        out_path = os.path.join(GENERATED_DIR, out_name)
        img.save(out_path)
        return out_path, params, agg

# ---------------------------
# GUI: tkinter
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Image Tracer & Brain Generator")
        root.geometry("420x160")
        Label(root, text="画像トレース -> brain_data.txt に保存\nbrain から自律生成", justify="left").pack(pady=8)

        self.trace_btn = Button(root, text="画像を選択してトレース", width=30, command=self.trace_image)
        self.trace_btn.pack(pady=6)

        self.gen_btn = Button(root, text="脳から生成", width=30, command=self.generate_from_brain)
        self.gen_btn.pack(pady=6)

        self.status = Label(root, text="status: ready", anchor="w")
        self.status.pack(fill="x", padx=8, pady=6)

    def trace_image(self):
        path = filedialog.askopenfilename(filetypes=[("PNG images","*.png"),("All files","*.*")])
        if not path:
            return
        try:
            tracer = Tracer(path)
            data = tracer.extract_all()
            # 追加情報: 簡易な説明テキスト（雰囲気の説明）を生成
            mood = data["mood"]
            mood_desc = []
            if mood["warmth"] > 0.05:
                mood_desc.append("warm")
            elif mood["warmth"] < -0.05:
                mood_desc.append("cool")
            else:
                mood_desc.append("neutral")
            if mood["complexity"] > 0.4:
                mood_desc.append("complex")
            else:
                mood_desc.append("simple")
            if mood["contrast"] > 0.3:
                mood_desc.append("high-contrast")
            else:
                mood_desc.append("soft-contrast")
            data["mood_text"] = " ".join(mood_desc)
            save_brain_entry(data)
            self.status.config(text=f"status: traced and saved ({os.path.basename(path)})")
            messagebox.showinfo("Traced", f"トレース完了: {os.path.basename(path)}\n保存先: {BRAIN_FILE}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_from_brain(self):
        try:
            gen = Generator(width=512, height=512)
            out_path, params, agg = gen.generate()
            self.status.config(text=f"status: generated {os.path.basename(out_path)}")
            messagebox.showinfo("Generated", f"生成完了: {out_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------------------------
# CLI補助: コマンドラインからも利用可能
# ---------------------------
def cli_trace(path):
    tracer = Tracer(path)
    data = tracer.extract_all()
    mood = data["mood"]
    mood_desc = []
    if mood["warmth"] > 0.05:
        mood_desc.append("warm")
    elif mood["warmth"] < -0.05:
        mood_desc.append("cool")
    else:
        mood_desc.append("neutral")
    if mood["complexity"] > 0.4:
        mood_desc.append("complex")
    else:
        mood_desc.append("simple")
    if mood["contrast"] > 0.3:
        mood_desc.append("high-contrast")
    else:
        mood_desc.append("soft-contrast")
    data["mood_text"] = " ".join(mood_desc)
    save_brain_entry(data)
    print("Traced and saved:", data["source"])

def cli_generate(out_name=None):
    gen = Generator(width=512, height=512)
    out_path, params, agg = gen.generate(out_name=out_name)
    print("Generated:", out_path)
    print("Params:", params)
    print("Agg:", agg)

# ---------------------------
# エントリポイント
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
        if cmd == "trace" and len(sys.argv) >= 3:
            cli_trace(sys.argv[2])
        elif cmd == "generate":
            out = sys.argv[2] if len(sys.argv) >= 3 else None
            cli_generate(out)
        else:
            print("Usage:")
            print("  python main.py                # GUI mode")
            print("  python main.py trace path.png")
            print("  python main.py generate [out.png]")
    else:
        root = Tk()
        app = App(root)
        root.mainloop()
