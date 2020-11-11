# DuckieRoadNet

This is the spcn_hackthon_2020 solvation of **AYSAT** team for predicting duckietown road types. 

Road types explained here: https://docs.duckietown.org/DT19/opmanual_duckietown/out.pdf#page=4

## Product Value 

#### Done

- [x] Fast and simple road classification model
- [x] Simple markup tool (python script)
- [x] Pretrained models for testing
- [x] Model predictions are better, than random dice roll
- [x] Good model prediction accuracy ~ 82-84%

#### To do

- [ ] Unbalanced dataset -> some classes solution detects with high acc, some with bad
- [ ] Unclear rule of classification, what is the next tile?
- [ ] Small Dataset for classification  

## Demonstration

Algorithm working on video from duckietown server: https://youtu.be/Ve79IPrYC6c

Model makes base prediction -- next tile class. Also, we print second-prioritized predicted class, cause our dataset not balanced && trained mostly on "straight line" tiles. 

![](./demo/demo_marked.gif)

![](./demo/demo1.gif)

## Run

Craete python venv, install requirements && download demo videos:
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 generate_dataset.py 
```

For launch prediction algorithm on your video:
```
python3 predict_roadtype.py -i ./video.mp4 -o ./out.mp4 --canny --cpu
-i -- path input
-o -- path output video
--canny -- outout video processed with canny
--cpu -- take cpu-trained model
```

### Create your own dataset

If you want landmark video for your own dataset (we hope you want, because we don't), use this script
(**doesn't needed for demo**) Launch landmarking script:
```
cd src
python3 markup_dataset.py -p ./data/video1/ 
// -p <relative path to data directory, that contains imgs>
// -o <JSON markup filename>
``` 

## HackathonTeam 

* [Gosudarkin Yaroslav](https://github.com/DarkFlink)
* [Gavrilov Andrey](https://github.com/AndrewGavril)
* [Glazunov Sergey](https://github.com/light5551)
* [Tokarev Andrey](https://github.com/yawningstudent)
* [Gizzatov Amir](https://github.com/gizzatovamir)