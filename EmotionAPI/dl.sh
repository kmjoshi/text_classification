# can't copy .env files to heroku
# https://unix.stackexchange.com/questions/223734/how-to-download-files-and-folders-from-onedrive-using-wget
# . .env

wget --no-check-certificate --content-disposition "https://onedrive.live.com/download?cid=C9414C674166322A&resid=C9414C674166322A%213453&authkey=AC_LMUh47nvkWyw" -O ./EmotionAPI/
wget --no-check-certificate --content-disposition "https://onedrive.live.com/download?cid=C9414C674166322A&resid=C9414C674166322A%213454&authkey=AJB04pAaqxQ0-wc" -O ./EmotionAPI/