#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import threading
from datetime import datetime
from collections import Counter

from PIL import Image, ImageOps, ImageFilter
import numpy as np
from noise import pnoise2

# ---------------------------
# パス解決: exeと同じ階層に保存
# ---------------------------
if getattr(sys, "frozen", False):
    # exe実行時はexeがあるディレクトリ
    executable_dir = os.path.dirname(sys.executable)
else:
    # スクリプト実行時はスクリプトがあるディレクトリ
    executable_dir = os.path.dirname(os.path.abspath(__file__))

BRAIN_FILE = os.path.join(executable_dir, "brain_data.txt")
# 生成物の出力先（必要なら）
OUTPUT_DIR = os.path.join(executable_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Tracer: 画像から特徴抽出
# ---------------------------
class Tracer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(image_path).convert("RGB")

    def extract_all(self):
        arr = np.asarray(self.img).astype(np.float32) / 255.0
        avg = (arr.reshape(-1, 3).mean(axis=0) * 255).astype(int).tolist()
        
        # 簡易的なドミナントカラー抽出
        flat_arr = (arr * 255).astype(np.uint8).reshape(-1, 3)
        # サンプリングして高速化
        sample_idx = np.random.choice(len(flat_arr), min(1000, len(flat_arr)), replace=False)
        sample_colors = [tuple(c) for c in flat_arr[sample_idx]]
        common_colors = [list(c) for c, _ in Counter(sample_colors).most_common(3)]

        # ムードベクトル（VSCodeなら青系の数値が出るはず）
        brightness = float(arr.mean())
        warmth = float(avg[0] - avg[2]) / 255.0 # R - B
        
        return {
            "source": os.path.basename(self.image_path),
            "timestamp": datetime.now().isoformat(),
            "avg_color": avg,
            "dominant_colors": common_colors,
            "mood": {"warmth": warmth, "contrast": brightness, "complexity": 0.5}
        }

# ---------------------------
# Generator: 記憶から画像を生成
# ---------------------------
class Generator:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    def generate_from_brain(self):
        # 記憶の読み込み
        entries = []
        if os.path.exists(BRAIN_FILE):
            with open(BRAIN_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try: entries.append(json.loads(line))
                    except: continue

        if not entries:
            print("記憶がありません。先に画像をトレースしてください。")
            return None

        # 過去の記憶を統合（平均化）
        avg_r = int(np.mean([e["avg_color"][0] for e in entries]))
        avg_g = int(np.mean([e["avg_color"][1] for e in entries]))
        avg_b = int(np.mean([e["avg_color"][2] for e in entries]))
        
        # パレット作成
        palette = []
        for e in entries:
            palette.extend(e["dominant_colors"])
        
        # --- 画像生成ロジック (Perlin Noise) ---
        scale = 100.0
        octaves = 6
        seed = random.random()
        
        img_arr = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # ノイズベースで色を塗る（真っ黒回避のためのスケーリング）
        for y in range(self.height):
            for x in range(self.width):
                n = pnoise2(x/scale, y/scale, octaves=octaves, base=int(seed*100))
                # nは -1.0 ~ 1.0 なので 0.0 ~ 1.0 に変換
                val = (n + 1.0) / 2.0
                
                # 記憶の色を合成（VSCodeの青をベースにするなど）
                img_arr[y, x, 0] = (avg_r / 255.0) * val
                img_arr[y, x, 1] = (avg_g / 255.0) * val
                img_arr[y, x, 2] = (avg_b / 255.0) * (1.0 - val * 0.2)

        # 0-255のuint8に変換（これが抜けると真っ黒になる）
        final_img = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(final_img)

# ---------------------------
# メイン処理
# ---------------------------
def main():
    print(f"--- 記憶ファイル: {BRAIN_FILE} ---")
    
    # モード選択（本来はGUIのボタン等で制御）
    mode = input("1: トレース(学習) 2: 生成(記憶から) > ")

    if mode == "1":
        path = input("画像パスを入力してください: ").strip('"')
        if os.path.exists(path):
            tracer = Tracer(path)
            data = tracer.extract_all()
            # 脳に書き込み（追記）
            with open(BRAIN_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
            print(f"学習完了: {data['source']} の特徴を覚えました。")
        else:
            print("ファイルが見つかりません。")

    elif mode == "2":
        gen = Generator()
        img = gen.generate_from_brain()
        if img:
            out_path = os.path.join(OUTPUT_DIR, f"gen_{int(time.time())}.png")
            img.save(out_path)
            img.show()
            print(f"生成完了: {out_path}")

if __name__ == "__main__":
    main()
