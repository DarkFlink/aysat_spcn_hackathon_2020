# DuckieRoadNet

This is the spcn_hackthon_2020 solvation of **AYSAT** team for predicting duckietown road types. 

Road types explained here: https://docs.duckietown.org/DT19/opmanual_duckietown/out.pdf#page=4

## Demonstration

Algorithm working on video from duckietown server: https://youtu.be/Ve79IPrYC6c

Model makes base prediction -- next tile class. Also, we print second-prioritized predicted class.

## Run

Install requirements && download demo videos:
```
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