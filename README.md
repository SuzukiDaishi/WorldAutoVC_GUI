# worldautovc-gui

WorldAutoVCのGUI版です．  
  
**MacOS only**

## インストール

先に`Rye`をインストールしておく必要があります  
[`Rye`のインストール方法](https://rye-up.com/guide/installation/)

```
# クローン
git clone https://github.com/SuzukiDaishi/WorldAutoVC_GUI.git
cd WorldAutoVC_GUI

# WorldAutoVCの重みをダウンロード
FILE_ID=1OfRQf3aBqz0PgMLrKUacxaWieVX_YG1E
FILE_NAME=world_autovc_jp_step001800.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
mv ./world_autovc_jp_step001800.pth ./models/world_autovc_jp_step001800.pth

# 入っていない場合
brew install portaudio

# アプリの準備
sudo rye sync --no-lock
```

### 動画解説

https://youtu.be/CA4KSlq-Wkk?si=ezDmdvy6Qp3G1d5o

## 使い方

```
cd WorldAutoVC_GUI
rye run app 
```

### 動画解説

https://youtu.be/buL3xYj2gr8
