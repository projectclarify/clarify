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

// Package sample controller provides a Kubernetes controller for a Caffe2 job resource.
package sample

/*
	//"errors"
	//"fmt"
	//"strings"
	//"wait"
	//"time"
	//util "github.com/projectclarify/clarify/util"
	//cbtutil "github.com/projectclarify/clarify/util/cbtutil"
*/

import (
	//#include "samplerConfig.h"
	"C"
	"fmt"
	"math/rand"
	"strconv"
	"sync"

	cbtutil "../util/cbtutil"
	bigtable "cloud.google.com/go/bigtable"
)

//SamplerConfig to be used by external go files
type SamplerConfig C.SamplerConfig

type sampler struct {
	config        SamplerConfig
	arrayToSample []int
}

type sampleJob struct {
	sampleIndex int
	jobID       string
}

var wg sync.WaitGroup
var summer chan int
var tbl *bigtable.Table

// RunSampler to be called from python with config
//export RunSampler
func RunSampler(config SamplerConfig) {
	sampler := newSampler(config)
	summer = make(chan int)
	sampler.Run(10)
}

// NewSamplerConfig returns sampler config with correct values, needed for calls from Go
func NewSamplerConfig(seed int, count int) SamplerConfig {
	return SamplerConfig{C.long(seed), C.long(count)}
}

func newSampler(config SamplerConfig) *sampler {

	sampler := &sampler{config, []int{1, 2, 3, 4, 5, 6, 7, 8}}

	return sampler
}

// Run will start up workers to perform sampling based on provided config
func (s *sampler) Run(threadiness int) error {

	//glog.Info("Starting hc/sampler")

	// For illustrative purposes and to check the gazelle / bazel setup.
	//util.dummyUtil()

	//glog.Infof("Starting %v workers", threadiness)

	tbl = cbtutil.NewConnection()
	jobChan := make(chan sampleJob, s.config.JobCount)

	for i := 0; i < threadiness; i++ {
		wg.Add(1)
		go s.worker(jobChan)
	}

	source := rand.NewSource(int64(s.config.RngSeed))
	rng := rand.New(source)

	var i C.long
	for i = 0; i < s.config.JobCount; i++ {
		jobChan <- sampleJob{rng.Intn(len(s.arrayToSample)), "job" + strconv.Itoa(int(i))}
	}

	//glog.Info("Started workers")
	sum := 0
	for i = 0; i < s.config.JobCount; i++ {
		sum += <-summer
	}
	close(jobChan)
	wg.Wait()
	fmt.Println("Real Sum:" + strconv.Itoa(sum))
	btSum := 0
	for i = 0; i < s.config.JobCount; i++ {
		btSum += cbtutil.ReadVal(tbl, "job"+strconv.Itoa(int(i)))
	}
	fmt.Println("Big Table Sum:" + strconv.Itoa(btSum))
	//glog.Info("Shutting down workers")

	return nil

}

func (s *sampler) worker(jobChan <-chan sampleJob) {
	defer wg.Done()

	for job := range jobChan {
		s.processJob(job)
	}
}

// processNextWorkItem will perform a unit of work
func (s *sampler) processJob(job sampleJob) bool {
	val := s.arrayToSample[job.sampleIndex]
	cbtutil.WriteVal(tbl, job.jobID, val)
	summer <- val
	return true
}
