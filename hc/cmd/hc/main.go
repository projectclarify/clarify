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

package main

//sampler "github.com/projectclarify/clarify/hc/pkg/sampler/sample"
import (
	"C"
	"flag"

	sampler "../../pkg/sampler"
)

func main() {
	seed := flag.Int("seed", 42, "RNG Seed")
	count := flag.Int("count", 10, "Job Count")
	flag.Parse()

	var config sampler.SamplerConfig
	config = sampler.NewSamplerConfig(*seed, *count)

	sampler.RunSampler(config)
}
