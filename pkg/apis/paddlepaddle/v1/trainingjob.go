package v1

import (
	"encoding/json"
	"fmt"
)

// Elastic returns true if the job can scale to more workers.
func (s *TrainingJob) Elastic() bool {
	return s.Spec.Trainer.MinInstance < s.Spec.Trainer.MaxInstance
}

// GPU convert Resource Limit Quantity to int
func (s *TrainingJob) GPU() int {
	q := s.Spec.Trainer.Resources.Limits.NvidiaGPU()
	gpu, ok := q.AsInt64()
	if !ok {
		// FIXME: treat errors
		gpu = 0
	}
	return int(gpu)
}

// NeedGPU returns true if the job need GPU resource to run.
func (s *TrainingJob) NeedGPU() bool {
	return s.GPU() > 0
}

func (s *TrainingJob) String() string {
	b, _ := json.MarshalIndent(s, "", "   ")
	return fmt.Sprintf("%s", b)
}
