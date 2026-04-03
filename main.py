#!/usr/bin/env python3
"""
main.py
GUI専用: 画像トレース -> brain_data.txt 保存 -> brainから自律生成
トレース時に1回生成、裏で100回学習（daemon thread）
PyInstaller バンドル対応（sys._MEIPASS）
"""

import os
import sys
import json
import time
import random
import threading
from datetime import datetime
from collections import Counter

# 画像処理・数値計算・ノイズ
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from noise import pnoise2

# ---------------------------
# PyInstaller 対応: バンドル内リソースの base path 解決
# ---------------------------
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

BRAIN_FILE = os.path.join(base_path, "brain_data.txt")
# GENERATED_DIR はもはや自動出力先に使わないがフォルダは保持
GENERATED_DIR = os.path.join(base_path, "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# ---------------------------
# デフォルトテンプレート
# ---------------------------
DEFAULT_BRAIN_TEMPLATE = {
    "avg_color": [128, 128, 128],
    "dominant_colors": [[128, 128, 128]],
    "mood": {"warmth": 0.0, "contrast": 0.5, "complexity": 0.2}
}

# ---------------------------
# ユーティリティ
# ---------------------------
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def _is_valid_brain_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if "avg_color" not in entry or "dominant_colors" not in entry or "mood" not in entry:
        return False
    try:
        _ = list(entry["avg_color"])
        _ = list(entry["dominant_colors"])
        _ = float(entry["mood"].get("warmth", 0.0))
    except Exception:
        return False
    return True

def save_brain_entry(entry: dict):
    e = dict(entry)
    if "avg_color" not in e:
        e["avg_color"] = DEFAULT_BRAIN_TEMPLATE["avg_color"]
    if "dominant_colors" not in e:
        e["dominant_colors"] = DEFAULT_BRAIN_TEMPLATE["dominant_colors"]
    if "mood" not in e:
        e["mood"] = DEFAULT_BRAIN_TEMPLATE["mood"]
    if "timestamp" not in e:
        e["timestamp"] = now_ts()
    if "source" not in e:
        e["source"] = "unknown"
    with open(BRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

def load_brain_entries():
    if not os.path.exists(BRAIN_FILE):
        return []
    entries = []
    with open(BRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if _is_valid_brain_entry(obj):
                entries.append(obj)
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
        lum = (0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2])
        return {
            "mean": float(lum.mean()),
            "var": float(lum.var()),
            "min": float(lum.min()),
            "max": float(lum.max())
        }

    def edge_density(self):
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
        avg = np.array(self.average_color()) / 255.0
        brightness = self.brightness_stats()
        warmth = float(avg[0] - avg[2])
        contrast = float(brightness["max"] - brightness["min"])
        complexity = float(self.edge_density())
        return {"warmth": warmth, "contrast": contrast, "complexity": complexity}

    def extract_all(self):
        data = {
            "source": os.path.basename(self.image_path),
            "timestamp": now_ts(),
            "avg_color": self.average_color(),
            "color_variance": self.color_variance(),
            "brightness": self.brightness_stats(),
            "edge_density": self.edge_density(),
            "dominant_colors": self.dominant_colors(n=3),
            "mood": self.mood_vector()
        }
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
        return data

# ---------------------------
# Generator: brainから読み出して生成
# ---------------------------
class Generator:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    def _aggregate_brain(self, entries):
        valid = [e for e in entries if _is_valid_brain_entry(e)]
        if not valid:
            return {
                "avg_color": DEFAULT_BRAIN_TEMPLATE["avg_color"],
                "palette": DEFAULT_BRAIN_TEMPLATE["dominant_colors"],
                "mood": DEFAULT_BRAIN_TEMPLATE["mood"]
            }
        avg_colors = np.array([e["avg_color"] for e in valid], dtype=float)
        avg_color = np.mean(avg_colors, axis=0).astype(int).tolist()
        palette = []
        for e in valid:
            for c in e.get("dominant_colors", []):
                try:
                    palette.append(tuple(int(x) for x in c))
                except Exception:
                    continue
        cnt = Counter(palette)
        most = [list(c) for c,_ in cnt.most_common(5)]
        if not most:
            most = [avg_color]
        moods = np.array([[e.get("mood",{}).get("warmth",0.0),
                           e.get("mood",{}).get("contrast",0.0),
                           e.get("mood",{}).get("complexity",0.0)] for e in valid], dtype=float)
        mood_avg = moods.mean(axis=0).tolist()
        return {
            "avg_color": avg_color,
            "palette": most,
            "mood": {"warmth": float(mood_avg[0]), "contrast": float(mood_avg[1]), "complexity": float(mood_avg[2])}
        }

    def _choose_params(self, agg):
        base_seed = int((agg["mood"]["warmth"]*1000 + agg["mood"]["contrast"]*100 + agg["mood"]["complexity"]*10000)) & 0xffffffff
        seed = (base_seed ^ int(time.time())) & 0xffffffff
        palette = agg.get("palette", [agg["avg_color"]])
        palette_f = [[c/255.0 for c in p] for p in palette]
        return {"seed": seed, "palette": palette_f, "mood": agg["mood"]}

    def _perlin_map(self, seed, scale=6.0, octaves=4, persistence=0.5, lacunarity=2.0):
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
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        return arr

    def _map_to_palette(self, noise_map, palette):
        h, w = noise_map.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        n_levels = len(palette)
        if n_levels == 0:
            palette = [[0.5,0.5,0.5]]
            n_levels = 1
        thresholds = np.linspace(0.0, 1.0, n_levels+1)
        for i in range(n_levels):
            mask = (noise_map >= thresholds[i]) & (noise_map < thresholds[i+1])
            col = np.array(palette[i]) * 255.0
            img[mask] = col.astype(np.uint8)
        return Image.fromarray(img)

    def generate_to_path(self, out_path, width=None, height=None):
        if width: self.width = width
        if height: self.height = height
        entries = load_brain_entries()
        agg = self._aggregate_brain(entries)
        params = self._choose_params(agg)

        mood = params["mood"]
        complexity = max(0.1, min(1.0, mood.get("complexity", 0.2)))
        contrast = max(0.1, min(1.0, mood.get("contrast", 0.5)))
        warmth = mood.get("warmth", 0.0)

        scale = 3.0 + complexity * 10.0
        octaves = int(1 + complexity * 5)
        persistence = 0.4 + (1.0 - complexity) * 0.3
        lacunarity = 1.8 + complexity * 1.5

        noise_map = self._perlin_map(seed=params["seed"], scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

        palette = params["palette"]
        if len(palette) < 3:
            avg = np.array(agg["avg_color"]) / 255.0
            while len(palette) < 3:
                palette.append(avg.tolist())

        def bias_color(c):
            c = np.array(c)
            c[0] = np.clip(c[0] + warmth*0.1, 0.0, 1.0)
            c = (c - 0.5) * (1.0 + contrast) + 0.5
            return np.clip(c, 0.0, 1.0).tolist()

        palette = [bias_color(c) for c in palette[:5]]

        img = self._map_to_palette(noise_map, palette)

        if complexity < 0.3:
            img = img.filter(ImageFilter.SMOOTH)
        elif complexity > 0.7:
            img = img.filter(ImageFilter.DETAIL)

        img.save(out_path)
        return out_path, params, agg

# ---------------------------
# バックグラウンド学習（トレース後に100回）
# ---------------------------
def background_learning_simulation(base_entry: dict, iterations: int = 100, delay: float = 0.01):
    def worker():
        for i in range(iterations):
            e = dict(base_entry)
            # small random perturbations to mood and palette to simulate learning
            mood = dict(e.get("mood", DEFAULT_BRAIN_TEMPLATE["mood"]))
            mood["warmth"] = mood.get("warmth", 0.0) + random.uniform(-0.02, 0.02)
            mood["contrast"] = max(0.0, mood.get("contrast", 0.0) + random.uniform(-0.02, 0.02))
            mood["complexity"] = max(0.0, mood.get("complexity", 0.0) + random.uniform(-0.02, 0.02))
            e["mood"] = mood
            # slightly mutate dominant colors
            new_palette = []
            for c in e.get("dominant_colors", []):
                try:
                    nc = [int(max(0, min(255, x + random.randint(-6, 6)))) for x in c]
                    new_palette.append(nc)
                except Exception:
                    new_palette.append(c)
            e["dominant_colors"] = new_palette
            e["timestamp"] = now_ts()
            e["source"] = f"simulated_learning_{i}"
            save_brain_entry(e)
            time.sleep(delay)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

# ---------------------------
# GUI: tkinter をここでインポート（ヘッドレス環境での import を安全にするため）
# ---------------------------
def launch_gui():
    from tkinter import Tk, Button, Label, filedialog, messagebox

    class App:
        def __init__(self, root):
            self.root = root
            root.title("neuro-trace-gen — Image Tracer & Brain Generator")
            root.geometry("480x200")
            Label(root, text="画像トレース -> brain_data.txt に保存\nトレース時に1回生成、裏で100回学習（非同期）", justify="left").pack(pady=8)

            self.trace_btn = Button(root, text="画像を選択してトレース", width=36, command=self.trace_image)
            self.trace_btn.pack(pady=6)

            self.gen_btn = Button(root, text="脳から生成（保存先を選択）", width=36, command=self.generate_from_brain)
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
                save_brain_entry(data)
                # トレース直後に1回だけ生成: 保存先をユーザーに選ばせる
                save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], initialfile=f"gen_{int(time.time())}.png")
                if save_path:
                    gen = Generator(width=512, height=512)
                    out_path, params, agg = gen.generate_to_path(save_path)
                    messagebox.showinfo("Generated", f"トレース後の1回生成を保存しました:\n{out_path}")
                else:
                    messagebox.showinfo("Traced", f"トレース完了（生成はスキップ）:\n{os.path.basename(path)}")

                # 裏で100回学習を開始（非同期）
                background_learning_simulation(data, iterations=100, delay=0.01)
                self.status.config(text=f"status: traced and learning started ({os.path.basename(path)})")
            except Exception as e:
                messagebox.showerror("Error", f"トレース中にエラーが発生しました:\n{e}")

        def generate_from_brain(self):
            try:
                save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], initialfile=f"gen_{int(time.time())}.png")
                if not save_path:
                    return
                gen = Generator(width=512, height=512)
                out_path, params, agg = gen.generate_to_path(save_path)
                self.status.config(text=f"status: generated {os.path.basename(out_path)}")
                messagebox.showinfo("Generated", f"生成完了: {out_path}")
            except Exception as e:
                messagebox.showerror("Error", f"生成中にエラーが発生しました:\n{e}")

    root = Tk()
    app = App(root)
    root.mainloop()

# ---------------------------
# エントリポイント（GUI専用）
# ---------------------------
if __name__ == "__main__":
    # brain file がなければ空ファイルを作成（無効な JSON は書き込まない）
    if not os.path.exists(BRAIN_FILE):
        open(BRAIN_FILE, "a", encoding="utf-8").close()

    # 常に GUI を起動（CLI は無効）
    launch_gui()
