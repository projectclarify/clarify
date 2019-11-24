
import os
import tensorflow as tf

dirname = "/home/chris_w_beitel/pcml-app/src/components"
query = "shop-products.js"

for filename in tf.gfile.ListDirectory(dirname):
  fpath = os.path.join(dirname, filename)
  with tf.gfile.Open(fpath, "r") as f:
    for line in f:
      if query in line:
        print(filename)
