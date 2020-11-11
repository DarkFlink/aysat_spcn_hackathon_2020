# DuckieRoadNet

This is the spcn_hackthon_2020 solvation for predicting duckietown road types. 

Road types explained here: https://docs.duckietown.org/DT19/opmanual_duckietown/out.pdf#page=4

## Run

Download demo videos:
```
python3 generate_dataset.py 
```

For launch prediction algorithm on your video:
```
python3 predict_roadtype.py -i ./video.mp4 -o ./out.mp4 --canny --cpu
-i -- path input
-o -- path output video
--canny -- outout video processed with canny
--cpu -- take cpu-trained model
>>>>>>> b97813b6cfa324ca8c91183c636274d20a9d73b4
```

## Additional

### Classification markup

Launch:
```
cd src
python3 markup_dataset.py -p ./data/video1/ 
// -p <relative path to data directory, that contains imgs>
// -o <JSON markup filename>
``` 

### HackathonTeam

* [Gosudarkin Yaroslav](https://github.com/DarkFlink)
* [Gavrilov Andrey](https://github.com/AndrewGavril)
* [Glazunov Sergey](https://github.com/light5551)
* [Tokarev Andrey](https://github.com/yawningstudent)
* [Gizzatov Amir](https://github.com/gizzatovamir)