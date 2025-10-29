# Visual Computing Assignment 2  
**Real-Time Image Processing Pipeline (CPU vs GPU)**  

---

This project implements a real-time video-processing pipeline in modern C++,
featuring two visual filters (**Pixelate** and **Sin City**) and interactive
geometric transformations (translation, rotation, scaling).
Both CPU and GPU versions are provided to benchmark and compare performance.

- **CPU path:** OpenCV filtering + `cv::warpAffine` for transforms  
- **GPU path:** OpenGL + GLSL fragment shaders on a fullscreen quad  
- **Modes:**  
  - `main.cpp` – automatically tests all configurations and exports FPS to CSV  
  - `interactive.cpp` – live camera / synthetic feed with real-time keyboard control  

---


Output:
Performance results in   `main.cpp` are saved as CSV files in the `build` directory:

`build/Debug/perf_summary_Debug.csv`

`build/Release/perf_summary_Release.csv`


---


Keyboard Controls in `interactive.cpp`:


| Key             | Function                                  |
| :-------------- | :---------------------------------------- |
| `G`             | Toggle GPU ↔ CPU mode                     |
| `1` / `2` / `3` | Select Filter — None / Pixelate / SinCity |
| `T`             | Toggle Transform (Affine) On / Off        |
| `↑` `↓` `←` `→` | Translate image (tx, ty)                  |
| `Q` / `E`       | Rotate image                              |
| `-` / `=`       | Zoom in / Zoom out                        |
| `Z` / `X`       | Adjust pixel block size (Pixelate filter) |
| `C` / `V`       | Adjust threshold (SinCity filter)         |
| `H`             | Show / Hide HUD help overlay              |
| `ESC`           | Quit program                              |
