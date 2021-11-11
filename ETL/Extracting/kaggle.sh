# install kaggle api to download dataset, download the json file
# from your kaggle account

! pip install -q kaggle

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

#check things are working
!kaggle datasets list


!kaggle datasets download 'user/dataset'

!unzip dataset.zip