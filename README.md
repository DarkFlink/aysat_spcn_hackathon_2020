# DuckieRoadNet

## Solvation




## Run

```
python3 predict_roadtype.py -i ./video.mp4 -o ./out.mp4 --canny --cpu
-i -- path input
-o -- path output video
--canny -- outout video processed with canny
--cpu -- take cpu-trained model
```


## Additional

### Generate dataset

Launch:
```
python3 generate_dataset.py 
```

### Classification markup

Launch:
```
cd src
python3 markup_dataset.py -p ./data/video1/ 
// -p <relative path to data directory, that contains imgs>
// -o <JSON markup filename>
``` 
