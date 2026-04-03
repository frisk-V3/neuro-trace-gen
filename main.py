import os
import sys
import json
import time
import random
import numpy as np
from PIL import Image
from noise import pnoise2
from collections import Counter
from datetime import datetime

# ---------------------------
# パス解決
# ---------------------------
if getattr(sys, "frozen", False):
    executable_dir = os.path.dirname(sys.executable)
else:
    executable_dir = os.path.dirname(os.path.abspath(__file__))

BRAIN_FILE = os.path.join(executable_dir, "brain_data.txt")
LAST_PATH_FILE = os.path.join(executable_dir, "last_image_path.txt")

# ---------------------------
# Generatorクラス (修正版)
# ---------------------------
class Generator:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    def generate_to_path(self, output_path):
        """外部から呼ばれるメインメソッド"""
        img = self.generate_from_brain()
        if img:
            img.save(output_path)
            return True
        return False

    def generate_from_brain(self):
        entries = []
        if os.path.exists(BRAIN_FILE):
            with open(BRAIN_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try: entries.append(json.loads(line))
                    except: continue
        
        if not entries:
            # 記憶がない場合のデフォルト色（VSCodeブルー系）
            avg_color = [0, 122, 204]
        else:
            # 過去の記憶から平均色を計算
            colors = np.array([e["avg_color"] for e in entries])
            avg_color = np.mean(colors, axis=0).astype(int).tolist()

        # 画像生成 (Perlin Noise)
        scale = 64.0
        seed = random.random()
        img_arr = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                n = pnoise2(x/scale, y/scale, octaves=6, base=int(seed*100))
                val = (n + 1.0) / 2.0
                # 色をのせる
                img_arr[y, x, 0] = (avg_color[0] / 255.0) * val
                img_arr[y, x, 1] = (avg_color[1] / 255.0) * val
                img_arr[y, x, 2] = (avg_color[2] / 255.0) * val

        # 重要：0-255にクリップしてuint8に変換（真っ暗回避）
        final_img = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(final_img)

# ---------------------------
# 実行ロジック
# ---------------------------
def run():
    # 前回使ったパスがあれば読み込む
    default_path = ""
    if os.path.exists(LAST_PATH_FILE):
        with open(LAST_PATH_FILE, "r", encoding="utf-8") as f:
            default_path = f.read().strip()

    print(f"現在の学習データ: {BRAIN_FILE}")
    img_path = input(f"画像パスを入力 (Enterで前回を使用: {default_path}): ").strip('" ')
    
    if not img_path:
        img_path = default_path

    if os.path.exists(img_path):
        # パスを保存
        with open(LAST_PATH_FILE, "w", encoding="utf-8") as f:
            f.write(img_path)
        
        # 本来はここでTracerを動かしてbrain_data.txtに追記する処理を入れる
        print(f"学習対象を固定しました: {img_path}")
        
        # テスト生成
        gen = Generator()
        out_name = f"output_{int(time.time())}.png"
        if gen.generate_to_path(out_name):
            print(f"生成成功: {out_name}")
    else:
        print("有効な画像パスを指定してください。")

if __name__ == "__main__":
    run()
