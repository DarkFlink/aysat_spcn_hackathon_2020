from utils import download, jpeg_from_mp4

url = 'http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20160430182226_quackmob.video.mp4'
filename = 'data.mp4'

download(url, filename)

jpeg_from_mp4(filename, 'data', 9)