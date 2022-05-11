## VoiceMOS Challenge data
1. 請不要外傳
2. 有兩個track，分別是main track跟OOD track
3. 下載完後請修改config.yaml裡面的dataset_folders到你指定的路徑
```
cd /path/to/put/dataset

wget 140.112.21.22:3600/main.tar.gz
wget 140.112.21.22:3600/ood.tar.gz

tar -zxvf main.tar.gz
tar -zxvf ood.tar.gz
```
