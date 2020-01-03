
## Golang datagen prototype

Enabling ~200x faster CBT to CBT datagen by making RPCs and other operations concurrent, operating on sharded audio, eliminating data augmentation (for now).

### Using current version

The current version is meant to be called from the `cbt_generate_and_load_examples_experimental` wrapper, like so:

```python

from pcml.utils import cbt_experimental as exp
from pcml.utils.cbt_utils import TFExampleSelection

example_selection = TFExampleSelection(project="project",
                                       instance="clarify",
                                       table="vox-celeb-single-frame-ex")

exp.cbt_generate_and_load_examples_experimental(
  project="clarify", problem_name="vox_celeb_single_frame",
  bigtable_instance="clarify", prefix="train",
  max_num_examples=101)

```

Which handles sampling of CBT raw table keys and passes these as a JSON-serialized command line argument to the go program.

### Using TFExample protos

Using TFExamples to serialize examples is a nice way to support various types and feature shapes in the same data structure; also our downstream steps support this format including with feature reading specs associated with Problems that track types and shapes. The following is the start of an attempt to construct serialized TFExamples in Golang. Where this fails, perhaps due to the developer not understanding enough Go, is when attempting to assign an Int64List to a field that expects types of the Feature iterface (of which this is supposed to be one).

For now the hacky solution was to modify feature.proto to only support a single feature type (BytesList) meaning that before the protos are built this patch of feature.proto needs to be copied into the tf codebase as described below.

The current version of the example generator (`main.go`) eschews TFExample serialization entirely opting for simply concatenating (labels, frames, audio) as one long uint8 byte array and requiring at training time / dataset construction time T2T.Problem's decode_example be modified to process this alternative serialization format.

#### Instructions

Set GOPATH if it isn't already:

```bash
GOPATH=$(go env GOPATH)
```

Obtain `protoc`:

```bash
PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local include/*
rm -f $PROTOC_ZIP
```

Obtain go dependencies for using protoc generated libraries:

```bash
go get github.com/golang/protobuf/proto
go get github.com/golang/protobuf/protoc-gen-go
```

Make sure `$GOPATH/bin` is on `$PATH`:

```bash
export PATH=$PATH:${GOPATH}/bin
```

Obtain tensorflow source into `./tmp`

```bash
git clone https://github.com/tensorflow/tensorflow
```

Copy feature.proto into `./tmp/tensorflow/tensorflow/core/example/`

Build the protos:

```bash
protoc \
  -I ./tmp/tensorflow \
  --go_out ./protos \
  ./tmp/tensorflow/tensorflow/core/example/*.proto

protoc \
  -I ./tmp/tensorflow \
  --go_out ./protos \
  ./tmp/tensorflow/tensorflow/core/framework/*.proto
```

Then use `proto.Marshal` to construct the `Example` object to feed:

```golang

import (
  example "github.com/tensorflow/tensorflow/tensorflow/go/core/example"
  proto "github.com/golang/protobuf/proto"
)

// ...

      ex := &example.Example{
        Features: &example.Features{
          Feature: map[string]*example.Feature{}}}

      ex.Features.Feature["audio"] = &example.Feature{Value: [][]byte{audio}}
      ex.Features.Feature["video"] = &example.Feature{Value: frameData}
      ex.Features.Feature["targets"] = &example.Feature{Value: [][]byte{{1}}}

      data, _ := proto.Marshal(ex)

```