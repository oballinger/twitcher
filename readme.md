
for app:
`export TFL_KEY= `
`node server.js & python detect.py & python -m http.server 8000 & wait`

for detection: 

`python collect.py --workers 100 --fps 1`

for training: 

1. Train — `python train_police.py --dataset ./dataset`
Takes your dataset/police/ folder as positives (~10 crops), samples negatives from car/truck/bus/motorcycle/. Uses MobileNetV2 with the backbone frozen — only trains a small binary head on top. Heavy augmentation (random affine, colour jitter, blur, perspective, erasing) stretches those 10 examples. Weighted sampler keeps batches balanced despite the class imbalance. Outputs police_classifier.pt.

2. Sweep — `python sweep.py --dataset ./dataset --model police_classifier.pt --threshold 0.5 --folders police`
Scores every crop in every folder, generates review.html where you click thumbnails to confirm real police hits, then export filenames. Run --confirm confirmed.txt to move them to police_confirmed/.
`python sweep.py --confirm confirmed.txt`

3. Retrain — move confirmed crops into police/, run train_police.py again with more data.
The police_detector.py is a drop-in replacement for your original — same is_police_vehicle(crop) interface, just loads the .pt model instead of running heuristics. Your detect.py and collect.py don't need any changes.
