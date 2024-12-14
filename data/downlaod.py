import gdown

url = 'https://drive.usercontent.google.com/download?id=1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK&export=download&authuser=0'
output = 'GOPRO_download.tgz'
gdown.download(url, output, quiet=False)