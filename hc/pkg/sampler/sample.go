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
	"fmt"
	"math/rand"
	"strconv"
	"sync"

	cbtutil "../util/cbtutil"
	bigtable "cloud.google.com/go/bigtable"
)

type samplerConfiguration struct {
	rngSeed  int
	jobCount int
}

type sampler struct {
	config        samplerConfiguration
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
func RunSampler() error {
	config := samplerConfiguration{42, 10}
	sampler := newSampler(config)
	summer = make(chan int)
	return sampler.Run(10)
}

func newSampler(config samplerConfiguration) *sampler {

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
	jobChan := make(chan sampleJob, s.config.jobCount)

	for i := 0; i < threadiness; i++ {
		wg.Add(1)
		go s.worker(jobChan)
	}

	source := rand.NewSource(int64(s.config.rngSeed))
	rng := rand.New(source)
	for i := 0; i < s.config.jobCount; i++ {
		jobChan <- sampleJob{rng.Intn(len(s.arrayToSample)), "job" + strconv.Itoa(i)}
	}

	//glog.Info("Started workers")
	sum := 0
	for i := 0; i < s.config.jobCount; i++ {
		sum += <-summer
	}
	close(jobChan)
	wg.Wait()
	fmt.Println("Real Sum:" + strconv.Itoa(sum))
	btSum := 0
	for i := 0; i < s.config.jobCount; i++ {
		btSum += cbtutil.ReadVal(tbl, "job"+strconv.Itoa(i))
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
