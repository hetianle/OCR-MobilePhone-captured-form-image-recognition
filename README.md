# OCR-MobilePhone-captured-form-image-recognition
The software could extract form content (mainly digital numbers) from mobile phone captured photos. It could recognizes the location (columns and rows of the form) of each cell, thus we could restore the data structure of form and inport to database.

### Pipeline
1. Image Preprocessing (Image gray scale, edge enhancement, banarization)
2. Form area extraction.
3. Line detection (vertical and horizontal)
4. Key points detection and cells extraction (record the coulmns and rows of wach cell)
5. Content recognition using tensorflow (mainly digitis numbers recognition)
6. Inport into database



### Denpidencies
  ``python3.6``,``opencv-python``,``pillow``,``tensorflow``
  
### Usage
    cd ../../OCR-MobilePhone-captured-form-image-recognition
    python main.py
    
### Highlights
1. raw image
<p align="center">
  <img src="image/example.jpg" width=640>
</p>

2. edge enhancement
<p align="center">
  <img src="image/生物量margin_dense.png" width=640 >
</p>

2. bin image
<p align="center">
  <img src="image/bin.png" width=640>
</p>

3. Form area extraction
<p align="center">
  <img src="image/_table_area_gray.png" width=640>
</p>

4. lines detection
<p align="center">
  <img src="image/检测到的水平线_标记直线.png" width=310 >
  <img src="image/检测到的竖直线22.png" width=310 >
</p>

5. Key points detection
<p align="center">
  <img src="image/not_sim_featurepoint4.png" width=640>
</p>

6. Cells extraction
<p align="center">
  <img src="image/etc-cells.png" width=640>
</p>
