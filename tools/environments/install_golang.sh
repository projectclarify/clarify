# Install Golang

wget https://dl.google.com/go/go1.13.linux-amd64.tar.gz
tar -C /usr/local -xzvf go1.13.linux-amd64.tar.gz

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

#GOPATH=/root/go /usr/local/go/bin/go get cloud.google.com/go/bigtable...

