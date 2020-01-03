// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main


import (

  "strings"
  "strconv"
  "context"
  "flag"
  "fmt"
  "log"
  "math/rand"
  "time"
  "encoding/json"
  "cloud.google.com/go/bigtable"
  //example "github.com/tensorflow/tensorflow/tensorflow/go/core/example"
  //proto "github.com/golang/protobuf/proto"
  "io/ioutil"

  // Profiling
  //"runtime/pprof"
  //"os"
  "cloud.google.com/go/profiler"

)


const (
  columnFamilyName  = "tfexample"
  columnName        = "example"
)


var timestamp = bigtable.Time(time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC))


const charset = "abcdefghijklmnopqrstuvwxyz"


var seededRand *rand.Rand = rand.New(
  rand.NewSource(time.Now().UnixNano()))


func unlex(key string) int {

  lexCharset := "abcdefghij"
  reverseCharset := map[byte]int{}
  for i:=0; i<len(lexCharset); i++ {
    reverseCharset[lexCharset[i]] = i
  }

  splitArray := strings.Split(key, "_")
  suffix := splitArray[len(splitArray) - 1]

  sLen := len(suffix)
  idx := 0
  for i := 0; i < sLen; i++ {
    idxPartial := reverseCharset[suffix[i]]
    idx = idx*10 + idxPartial
  }

  return idx

}


func StringWithCharset(length int, charset string) string {

  b := make([]byte, length)

  for i := range b {
    b[i] = charset[seededRand.Intn(len(charset))]
  }

  return string(b)

}


func randomKey(prefix string, length int) string {

  suffix := StringWithCharset(length, charset)

  return fmt.Sprintf("%s_%s", prefix, suffix)

}


func sliceContains(list []string, target string) bool {

  for _, s := range list { if s == target {return true} }

  return false

}


type CorrespondenceSampleMeta struct {

  AudioKeys []string

  FrameKeys []string

  AudioSampleBounds []int

  Labels map[string]int

}


func closeClients(client *bigtable.Client,
                  adminClient *bigtable.AdminClient) {

  if err := client.Close(); err != nil {
      log.Fatalf("Could not close data operations client: %v", err)
  }

  if err := adminClient.Close(); err != nil {
      log.Fatalf("Could not close admin client: %v", err)
  }

}


func maybeCreateTable(ctx context.Context,
                      adminClient *bigtable.AdminClient,
                      targetTableName string,
                      columnFamilyName string) {
  
  tables, err := adminClient.Tables(ctx)
  if err != nil {log.Fatalf("Could not fetch table list: %v", err)}

  if !sliceContains(tables, targetTableName) {

      log.Printf("Creating table %s", targetTableName)
      if err := adminClient.CreateTable(ctx, targetTableName); err != nil {
          log.Fatalf("Could not create table %s: %v", targetTableName, err)
      }

  }

  tblInfo, err := adminClient.TableInfo(ctx, targetTableName)
  if err != nil {
      log.Fatalf("Could not read info for table %s: %v", targetTableName, err)
  }

  if !sliceContains(tblInfo.Families, columnFamilyName) {
    err := adminClient.CreateColumnFamily(ctx, targetTableName, columnFamilyName)
    if err != nil {
      log.Fatalf("Could not create column family %s: %v", columnFamilyName, err)
    }
  }

}


func ensureTable(ctx context.Context,
                 adminClient *bigtable.AdminClient,
                 tableName string) {

  tables, err := adminClient.Tables(ctx)
  if err != nil {log.Fatalf("Could not fetch table list: %v", err)}

  if !sliceContains(tables, tableName) {
      log.Fatalf("Source table does not exist: %s", tableName)
  }

}


func allocateMutationBuffer(batchSize int) ([]*bigtable.Mutation, []string) {

  mutations := make([]*bigtable.Mutation, batchSize)

  rowKeys := make([]string, batchSize)
  
  return mutations, rowKeys

}


func keyIdx(key string) int {
  key_split := strings.Split(key, "_")
  key_idx_str := key_split[len(key_split) - 1]
  key_idx, _ := strconv.Atoi(key_idx_str)
  return key_idx
}


func readAudio(ctx context.Context, table *bigtable.Table, keys []string,
               audioBounds []int, audioBlockSize int) []byte {

  // TODO: pass audio shard size as a parameter
  audioData := make([]byte, len(keys)*audioBlockSize)
  minKeyIdx := unlex(keys[0])

  rr := bigtable.NewRange(keys[0], keys[len(keys) - 1])

  err := table.ReadRows(ctx, rr, func(r bigtable.Row) bool {
    rowKey := r.Key()
    rowKeyIdx := unlex(rowKey)
    idxStart := rowKeyIdx - minKeyIdx
    start := idxStart*audioBlockSize
    end := (idxStart + 1)*audioBlockSize
    copy(audioData[start:end], r["audio"][0].Value)
    return true
  })
  if err != nil {
    panic(err)
  }

  return audioData[audioBounds[0]:audioBounds[1]]

}


func readContiguous(ctx context.Context, table *bigtable.Table, keys []string,
                    blockSize int, column string) []byte {

  data := make([]byte, len(keys)*blockSize)

  minKeyIdx := unlex(keys[0])

  rr := bigtable.NewRange(keys[0], keys[len(keys) - 1])

  err := table.ReadRows(ctx, rr, func(r bigtable.Row) bool {
    rowKey := r.Key()
    rowKeyIdx := unlex(rowKey)
    idxStart := rowKeyIdx - minKeyIdx
    start := idxStart*blockSize
    end := (idxStart + 1)*blockSize
    copy(data[start:end], r[column][0].Value)
    return true
  })
  if err != nil {
    panic(err)
  }

  return data

}


/*func readFrames(ctx context.Context, table *bigtable.Table, keys []string,
                frameXYDim int, frameNumChannels int) []byte {

  singleFrameSize := frameXYDim*frameXYDim*frameNumChannels
  frameData := make([]byte, len(keys)*singleFrameSize)

  for i, key := range keys {

    frame, _ := table.ReadRow(ctx, key)
    start := i*singleFrameSize
    end := (i+1)*singleFrameSize
    copy(frameData[start:end], frame["video_frames"][0].Value)

  }

  return frameData

}*/


func makeExampleMutation(data []byte) *bigtable.Mutation {

  mutation := bigtable.NewMutation()

  mutation.Set(columnFamilyName, columnName, timestamp, data)

  return mutation

}


func applyMutations(ctx context.Context,
                    targetTable *bigtable.Table,
                    keys []string,
                    mutations []*bigtable.Mutation,
                    inProgressQ chan int,
                    doneQ chan int) {

  rowErrs, err := targetTable.ApplyBulk(ctx, keys, mutations)
  if err != nil {
    log.Fatalf("Could not apply bulk row mutation: %v", err)
  }

  if rowErrs != nil {
    for _, rowErr := range rowErrs {
      log.Printf("Error writing row: %v", rowErr)
    }
    log.Fatalf("Could not write some rows")
  }

  work := <-inProgressQ
  doneQ <- work

}


func blockForNCompletions(doneQ chan int, numOutstandingBatches int) {
  /*

  Tick every 100ms.
  
  When work is completed, a message is received via the done queue. When
  the counter on work completed hits the target, returns.
  
  */

  tick := time.Tick(100 * time.Millisecond)
  epochs := 0

  for {
    
    select {
      case <- tick:
        epochs++
      case <- doneQ:
        numOutstandingBatches--
        if (numOutstandingBatches == 0) {
          return
        }
    }
  }
  
}


func buildAndSendBatch(ctx context.Context,
                       sourceTable *bigtable.Table,
                       targetTable *bigtable.Table,
                       mutationBatchSize int,
                       samples []CorrespondenceSampleMeta,
                       batchPrefix string,
                       inProgressQ chan int,
                       doneQ chan int,
                       audioShardSize int,
                       frameXYDim int,
                       frameNumChannels int) {

    mutations, rowKeys := allocateMutationBuffer(mutationBatchSize)

    for i := 0; i < mutationBatchSize; i++ {

      labels := []byte{byte(samples[i].Labels["overlap"]),
                       byte(samples[i].Labels["same_video"])}

      // Obtain the audio data
      audioDataTmp := readContiguous(ctx, sourceTable, samples[i].AudioKeys,
                                     audioShardSize, "audio")
      audioBounds := samples[i].AudioSampleBounds
      audio := audioDataTmp[audioBounds[0]:audioBounds[1]]

      // Obtain frame data
      videoBlockSize := frameXYDim*frameXYDim*frameNumChannels
      frameData := readContiguous(ctx, sourceTable, samples[i].FrameKeys,
                                  videoBlockSize, "video_frames")

      /*
      audio := readAudio(ctx, sourceTable,
                         samples[i].AudioKeys,
                         samples[i].AudioSampleBounds,
                         audioShardSize)

      //frameKeySubset := []string{samples[i].FrameKeys[0]}
      frameData := readFrames(ctx, sourceTable, samples[i].FrameKeys,
                              frameXYDim, frameNumChannels)
      */

      // ------
      // Pre-allocate array and write directly

      labelsLen := len(labels)
      //framesDataLen := len(frameData[0])
      framesDataLen := len(frameData)
      audioLen := len(audio)
      audioOffset := labelsLen + framesDataLen
      concatLength := labelsLen + framesDataLen + audioLen
      data := make([]byte, concatLength)
      copy(data[0:labelsLen], labels)
      copy(data[labelsLen:(labelsLen + framesDataLen)], frameData)
      copy(data[audioOffset:(audioOffset + audioLen)], audio)

      // ------
      // Include labels

      //labelsLen := len(labels)
      //concatLength := labelsLen + framesDataLen + audioLen
      //concatData := make([]byte, concatLength)
      //copy(concatData[0:labelsLen], labels)
      //copy(concatData[labelsLen:(labelsLen + framesDataLen)], frameData[0])
      //copy(concatData[audioOffset:(audioOffset + audioLen)], audio)

      // -------
      // Append

      //data := append(frameData[0], audio...)

      // -------

      mutation := bigtable.NewMutation()
      mutation.Set(columnFamilyName, columnName, timestamp, data)
      mutations[i] = mutation

      //mutations[i] = makeExampleMutation(append(frameData[0], audio...))

      // append(frameData[0], audio...)
      //(*mutations[i]).Set(columnFamilyName, columnName, timestamp, audio)

      //rowKeys[i] = randomKey(targetPrefix, targetKeySize)
      rowKeys[i] = fmt.Sprintf("%s%s", batchPrefix, string(charset[i]))
      //log.Printf("%s", rowKeys[i])

    }

    //<- inProgressQ

    // Async the application of a batch of mutations
    //inProgressQ <- 1

    //go applyMutations(ctx, targetTable, rowKeys, mutations, inProgressQ, doneQ)

    rowErrs, err := targetTable.ApplyBulk(ctx, rowKeys, mutations)
    if err != nil {
      log.Fatalf("Could not apply bulk row mutation: %v", err)
    }

    if rowErrs != nil {
      for _, rowErr := range rowErrs {
        log.Printf("Error writing row: %v", rowErr)
      }
      log.Fatalf("Could not write some rows")
    }

    work := <-inProgressQ
    doneQ <- work
  
}


func check(e error) {
    if e != nil {
        panic(e)
    }
}


func main() {

  project := flag.String("project", "", "The Google Cloud Platform project ID")
  instance := flag.String("instance", "", "The Google Cloud Bigtable instance ID")

  sourceTableName := flag.String("sourceTableName", "", "The table to sample")
  targetTableName := flag.String("targetTableName", "", "The table to populate")

  targetPrefix := flag.String("targetPrefix", "train", "Key prefix for target rows")
  targetKeySize := flag.Int("targetKeySize", 4, "Size of random key")

  sourceFrameX := flag.Int("sourceFrameX", 0, "Source frame x size")
  sourceFrameY := flag.Int("sourceFrameY", 0, "Source frame y size")
  sourceFrameC := flag.Int("sourceFrameC", 0, "Source frame num channels")

  targetFrameX := flag.Int("targetFrameX", 0, "Target frame x size")
  targetFrameY := flag.Int("targetFrameY", 0, "Target frame y size")
  targetFrameC := flag.Int("targetFrameC", 0, "Target frame num channels")

  audioShardSize := flag.Int("audioShardSize", 1000, "Audio shard size")

  samplesPath := flag.String("samplesPath", "", "JSON-encoded sample meta as a file")

  //cpuprofile := flag.String("cpuprofile", "", "write cpu profile to file")

  flag.Parse()

  requiredFlags := []string{"project", "instance"}

  for _, f := range requiredFlags {

    if flag.Lookup(f).Value.String() == "" {
      log.Fatalf("The %s flag is required.", f)
    }

  }

  log.Printf("Reading from table: %s", *sourceTableName)
  log.Printf("Populating table: %s", *targetTableName)

  log.Printf("Source frame shape: %d, %d, %d", *sourceFrameX, *sourceFrameY, *sourceFrameC)
  log.Printf("Target frame shape: %d, %d, %d", *targetFrameX, *targetFrameY, *targetFrameC)

  // Profiler initialization, best done as early as possible.
  if err := profiler.Start(profiler.Config{
          Service:        "myservice",
          ServiceVersion: "1.0.4",
          // ProjectID must be set if not running on GCP.
		  NoHeapProfiling:      true,
		  NoAllocProfiling:     true,
		  NoGoroutineProfiling: true,
		  DebugLogging:         true,
          ProjectID: "clarify",
  }); err != nil {
          // TODO: Handle error.
  }

  /*
  if *cpuprofile != "" {
    f, err := os.Create(*cpuprofile)
    if err != nil {
        log.Fatal(err)
    }
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()
  }
  */

  samplesJson, err := ioutil.ReadFile(*samplesPath)
  check(err)

  var samples []CorrespondenceSampleMeta
  json.Unmarshal([]byte(samplesJson), &samples)

  ctx := context.Background()
  
  log.Printf("Loaded correspondence sample metadata")

  adminClient, err := bigtable.NewAdminClient(ctx, *project, *instance)
  if err != nil {log.Fatalf("Could not create admin client: %v", err)}

  maybeCreateTable(ctx, adminClient, *targetTableName, columnFamilyName)

  ensureTable(ctx, adminClient, *sourceTableName)

  log.Printf("Finished ensuring table...")
  
  client, err := bigtable.NewClient(ctx, *project, *instance)
  if err != nil { log.Fatalf("Could not create client: %v", err) }

  sourceTable := client.Open(*sourceTableName)
  targetTable := client.Open(*targetTableName)

  numExamples := len(samples)
  log.Printf("Processing %d examples", numExamples)

  numConcurrentRPCs := 200
  mutationBatchSize := 26

  inProgressQ := make(chan int, numConcurrentRPCs)
  doneQ := make(chan int)

  numBatches := int(numExamples/mutationBatchSize)

  log.Printf("Processing %d batches", numBatches)

  for b := 0; b < numBatches; b++ {
    
    batchStart := b*mutationBatchSize
    batchEnd := (b+1)*mutationBatchSize
    batchPrefix := randomKey(*targetPrefix, *targetKeySize - 1)

    inProgressQ <- 1
    go buildAndSendBatch(ctx, sourceTable, targetTable,
                         mutationBatchSize,
                         samples[batchStart:batchEnd],
                         batchPrefix,
                         inProgressQ, doneQ,
                         *audioShardSize,
                         64, 1)

  }

  blockForNCompletions(doneQ, numBatches)

  closeClients(client, adminClient)

}
