---
title: CoBeta
emoji: 🧗‍♀️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🧗‍♀️ CoBeta: Computer Vision-Powered Climbing Beta Analysis

Advanced motion analysis system for climbing performance optimization using pose tracking and color-based hold detection.

## Features

- **Trajectory Tracking**: Track body parts (hands, feet, hips) and visualize movement paths
- **Route Segmentation**: Detect and segment climbing holds by color using YOLO and FastSAM
- **Color Detection**: Filter holds by specific colors (black, blue, brown, cream, green, orange, pink, purple, red, white, yellow)
- **Image & Video Processing**: Support for both image and video analysis

## How to Use

1. **Trajectory Tracking Tab**: Upload a video to track body part trajectories
2. **Route Segmentation Tab**: 
   - Upload an image or video
   - Select target colors to detect
   - Analyze climbing holds with color-based segmentation

## Model Files

- `best_color.pt`: YOLO model for color detection (should be in repo root or `src/beta/`)
- `FastSAM-x.pt`: FastSAM model for segmentation (auto-downloads if not present)

## Requirements

See `requirements.txt` for all dependencies.

