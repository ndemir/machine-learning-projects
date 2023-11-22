pip install --quiet git+https://github.com/huggingface/transformers.git
pip install accelerate
pip install bitsandbytes
pip install -q flash-attn --no-build-isolation

mkdir -p bin
mkdir -p files

wget https://github.com/ytdl-org/ytdl-nightly/releases/download/2023.09.25/youtube-dl -O bin/youtube-dl
chmod +x bin/youtube-dl

sudo apt-get install ffmpeg -y