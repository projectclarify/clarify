// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package controller provides a Kubernetes controller for a Caffe2 job resource.
package sample

import (

  "errors"
  "fmt"
  "strings"
  "time"
  
  "github.com/golang/glog"

  util "github.com/projectclarify/clarify/util"
  cbtutil "github.com/projectclarify/clarify/util/cbtutil"

)

type SamplerConfiguration struct {

}

type Sampler struct {

  config SamplerConfiguration

}

func New() (*Sampler, error) {

    sampler := &Sampler{
    }

    return sampler, nil

}

// Run will start up workers to perform sampling based on provided config
func (s *Sampler) Run(threadiness int, stopCh <-chan struct{}) error {

    glog.Info("Starting hc/sampler")

    // For illustrative purposes and to check the gazelle / bazel setup.
    util.dummyUtil()

    glog.Infof("Starting %v workers", threadiness)
    for i := 0; i < threadiness; i++ {
        go wait.Until(s.runWorker, time.Second, stopCh)
    }

    glog.Info("Started workers")
    <-stopCh
    glog.Info("Shutting down workers")

    return nil

}

// runWorker will continuously call processNextWorkItem
func (s *Sampler) runWorker() {
    for s.processNextWorkItem() {
    }
}

// processNextWorkItem will perform a unit of work
func (s *Sampler) processNextWorkItem() bool {
    return true
}