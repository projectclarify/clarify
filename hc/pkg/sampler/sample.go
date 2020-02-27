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

	util "../util"
	cbtutil "../util/cbtutil"
	bigtable "cloud.google.com/go/bigtable"
	proto "github.com/golang/protobuf/proto"
)

//SamplerConfig to be used by external go files
type SamplerConfig C.SamplerConfig

type sampler struct {
	config SamplerConfig
	video  util.VideoToRead
}

type samplingJob struct {
	frameIndex  int
	sampleIndex int
	jobID       string
}

var wg sync.WaitGroup
var frameChan chan *util.Frame

//var summer chan int
//var tbl *bigtable.Table

// RunSampler to be called from python with config
//export RunSampler
func RunSampler(config SamplerConfig) {
	tbl := cbtutil.NewConnection()
	setupDatabaseForTest(tbl)
	marsheledVideoInput := cbtutil.ReadVal(tbl, "0_video")
	videoInput := util.VideoToRead{}
	proto.Unmarshal(marsheledVideoInput, &videoInput)

	sampler := newSampler(config, videoInput)
	frameChan = make(chan *util.Frame)
	//summer = make(chan int)
	sampler.run(tbl, 2)
}

// NewSamplerConfig returns sampler config with correct values, needed for calls from Go
func NewSamplerConfig(seed int) SamplerConfig {
	return SamplerConfig{C.long(seed)}
}

func newSampler(config SamplerConfig, video util.VideoToRead) *sampler {

	sampler := &sampler{config, video}

	return sampler
}

// run will start up workers to perform sampling based on provided config
func (s *sampler) run(tbl *bigtable.Table, threadiness int) error {

	//glog.Info("Starting hc/sampler")

	// For illustrative purposes and to check the gazelle / bazel setup.
	//util.dummyUtil()

	//glog.Infof("Starting %v workers", threadiness)

	jobChan := make(chan samplingJob, s.video.FrameCount)

	for i := 0; i < threadiness; i++ {
		wg.Add(1)
		go s.worker(jobChan)
	}

	source := rand.NewSource(int64(s.config.RngSeed))
	rng := rand.New(source)

	for i := 0; i < int(s.video.FrameCount); i++ {
		jobChan <- samplingJob{i, rng.Intn(int(s.video.XRes * s.video.YRes)), "job" + strconv.Itoa(int(i))}
	}

	//glog.Info("Started workers")
	/*sum := 0
	for i = 0; i < s.config.JobCount; i++ {
		sum += <-summer
	}*/

	frames := []*util.Frame{}
	for i := 0; i < int(s.video.FrameCount); i++ {
		frames = append(frames, <-frameChan)
	}

	close(jobChan)
	wg.Wait()
	writeVid := util.VideoToWrite{Id: s.video.Id, FrameCount: s.video.FrameCount, XRes: s.video.XRes, YRes: s.video.YRes, Frames: frames}
	valToWrite, err := proto.Marshal(&writeVid)
	fmt.Println(proto.MarshalTextString(&writeVid))
	if err != nil {
		return err
	}
	cbtutil.WriteVal(tbl, string(writeVid.Id)+"_video", valToWrite)
	readVal := cbtutil.ReadVal(tbl, string(writeVid.Id)+"_video")
	unmarsheledReadVal := &util.VideoToWrite{}
	proto.Unmarshal(readVal, unmarsheledReadVal)
	fmt.Println(proto.MarshalTextString(unmarsheledReadVal))

	//fmt.Println("Real Sum:" + strconv.Itoa(sum))
	/*btSum := 0
	for i = 0; i < s.config.JobCount; i++ {
		btSum += cbtutil.ReadVal(tbl, "job"+strconv.Itoa(int(i)))
	}
	fmt.Println("Big Table Sum:" + strconv.Itoa(btSum))*/
	//glog.Info("Shutting down workers")

	return nil

}

func (s *sampler) worker(jobChan <-chan samplingJob) {
	defer wg.Done()

	for job := range jobChan {
		s.processJob(job)
	}
}

// processJob will perform a unit of work
func (s *sampler) processJob(job samplingJob) bool {
	frame := s.video.Frames[job.frameIndex]
	frame.Vals[job.sampleIndex] = frame.Vals[job.sampleIndex] + 1
	/*for idx, frameVal := range frame.Vals {
		frame.Vals[idx] = frameVal + 1
	}*/
	//cbtutil.WriteVal(tbl, job.jobID, val)
	//summer <- val
	frameChan <- frame
	return true
}

func setupDatabaseForTest(tbl *bigtable.Table) {
	frames := []*util.Frame{{Vals: []int32{1, 1, 1, 1}}, {Vals: []int32{2, 2, 2, 2}}, {Vals: []int32{3, 3, 3, 3}}, {Vals: []int32{4, 4, 4, 4}}, {Vals: []int32{5, 5, 5, 5}}}
	testVid := util.VideoToRead{Id: 0, FrameCount: 5, XRes: 2, YRes: 2, Frames: frames}
	val, err := proto.Marshal(&testVid)
	if err != nil {
		panic(err)
	}
	cbtutil.WriteVal(tbl, strconv.Itoa(int(testVid.Id))+"_video", val)
}
