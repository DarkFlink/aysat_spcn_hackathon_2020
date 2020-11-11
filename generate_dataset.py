from utils import process_video


process_video(url='http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20160430182226_quackmob.video.mp4', path=f'data/video1', filename='data1.mp4')
process_video(url='http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20160512194050_morty.video.mp4', path=f'data/video2', skip_rate=18, filename='data2.mp4')
process_video(url='http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20160429225147_neptunus.video.mp4', path=f'data/video3', skip_rate=5, filename='data3.mp4')
process_video(url='https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20160504165945_julie.video.mp4',path=f'data/video4',skip_rate=10,filename='data4.mp4')