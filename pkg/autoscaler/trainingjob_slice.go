package autoscaler

import 	padv1 "github.com/paddlepaddle/edl/pkg/apis/paddlepaddle/v1"

type trainingjobSlice []*padv1.TrainingJob

func (ts trainingjobSlice) Len() int {
	return len(ts)
}

func (ts trainingjobSlice) Swap(i, j int) {
	ts[i], ts[j] = ts[j], ts[i]
}

func (ts trainingjobSlice) Less(i, j int) bool {
	scoreA := ts[i].Fulfillment()
	scoreB := ts[j].Fulfillment()

	if scoreA == scoreB {
		resA := ts[j].Spec.Trainer.Resources
		resB := ts[j].Spec.Trainer.Resources
		resALimitsGPU := *resA.Limits.NvidiaGPU()
		resBLimitsGPU := *resB.Limits.NvidiaGPU()
		if resALimitsGPU.Cmp(resBLimitsGPU) == 0 {
			resARequestsCPU := *resA.Requests.Cpu()
			resBRequestsCPU := *resB.Requests.Cpu()
			if resARequestsCPU.Cmp(resBRequestsCPU) == 0 {
				resARequestsMem := *resA.Requests.Memory()
				resBRequestsMem := *resB.Requests.Memory()
				return resARequestsMem.Cmp(resBRequestsMem) == -1
			}
			return resARequestsCPU.Cmp(resBRequestsCPU) == -1
		}
		return resALimitsGPU.Cmp(resBLimitsGPU) == -1
	}
	return scoreA < scoreB
}