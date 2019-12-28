
apt-get update

apt-get install -y pkg-config \
                   zip \
                   g++ \
                   zlib1g-dev \
                   unzip \
                   ffmpeg \
                   wget \
                   curl \
                   libsm6 \
                   libxext6 \
                   libxrender-dev \
                   libglib2.0-0 \
                   htop \
                   git

pip install opencv-python==4.1.0.25

# Install Golang

wget https://dl.google.com/go/go1.13.linux-amd64.tar.gz
tar -C /usr/local -xzvf go1.13.linux-amd64.tar.gz

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

GOPATH=/root/go /usr/local/go/bin/go get cloud.google.com/go/bigtable...
