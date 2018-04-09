package autoscaler

import (
	log "github.com/golang/glog"
	padv1 "github.com/paddlepaddle/edl/pkg/apis/paddlepaddle/v1"
	"github.com/paddlepaddle/edl/pkg/updater"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"sort"
	"time"
	"sync"
)

const (
	defaultLoopDur = time.Second * 5
)

// Autoscaler launches and scales the training jobs.
type Autoscaler struct {
	// kubeCli is standard kubernetes client.
	KubeCli        kubernetes.Interface

	Jobupdater     *sync.Map

	MaxLoadDesired float64
}

// WithMaxLoadDesired init with maxLoadDesired
func WithMaxLoadDesired(maxLoadDesired float64) func(as *Autoscaler) {
	return func(as *Autoscaler) {
		as.MaxLoadDesired = maxLoadDesired
	}
}

// NewAutoscaler creates a new Autoscaler.
func NewAutoscaler(kubeClient kubernetes.Interface, Jobupdater *sync.Map, options ...func(*Autoscaler)) *Autoscaler {
	c := &Autoscaler{
		KubeCli:        kubeClient,
		Jobupdater:     Jobupdater,
		MaxLoadDesired: 1.0,
	}
	for _, option := range options {
		option(c)
	}
	return c
}

// InquiryResource returns the idle and total resources of the k8s cluster.
func (a *Autoscaler) InquiryResource() (ClusterResource, error) {

	nodes := a.KubeCli.CoreV1().Nodes()

	nodeList, err := nodes.List(metav1.ListOptions{})
	if err != nil {
		return ClusterResource{}, err
	}
	allocatable := make(v1.ResourceList)
	nodesCPUIdleMilli := make(map[string]int64)
	nodesMemoryFreeMega := make(map[string]int64)

	for _, node := range nodeList.Items {
		nodesCPUIdleMilli[node.GetObjectMeta().GetName()] =
			node.Status.Allocatable.Cpu().ScaledValue(resource.Milli)
		nodesMemoryFreeMega[node.GetObjectMeta().GetName()] =
			node.Status.Allocatable.Memory().ScaledValue(resource.Mega)
		AddResourceList(allocatable, node.Status.Allocatable)
	}

	// Get non-terminated pods from all namespaces.
	namespace := ""

	// FIXME(typhoonzero): scan all pods is not a efficient way.
	// NOTE: pending pods need to be caculated for scale down.
	// NOTE: "terminating" pods' status is still running, do not
	// scale up/down the job if job is still at last scaling
	// process.
	fieldSelector, err := fields.ParseSelector("status.phase!=" + string(api.PodSucceeded) + ",status.phase!=" + string(api.PodFailed))
	if err != nil {
		return ClusterResource{}, err
	}

	allPodsList, err := a.KubeCli.CoreV1().Pods(namespace).List(metav1.ListOptions{FieldSelector: fieldSelector.String()})
	if err != nil {
		return ClusterResource{}, err
	}

	allReqs, allLimits, err := getPodsTotalRequestsAndLimits(allPodsList)
	if err != nil {
		return ClusterResource{}, err
	}

	err = updateNodesIdleResource(allPodsList, nodesCPUIdleMilli, nodesMemoryFreeMega)
	if err != nil {
		return ClusterResource{}, err
	}

	res := ClusterResource{
		NodeCount:       len(nodeList.Items),
		GPUTotal:        int(allocatable.NvidiaGPU().Value()),
		CPUTotalMilli:   allocatable.Cpu().ScaledValue(resource.Milli),
		MemoryTotalMega: allocatable.Memory().ScaledValue(resource.Mega),

		GPURequest:        int(allReqs.NvidiaGPU().Value()),
		CPURequestMilli:   allReqs.Cpu().ScaledValue(resource.Milli),
		MemoryRequestMega: allReqs.Memory().ScaledValue(resource.Mega),

		GPULimit:        int(allLimits.NvidiaGPU().Value()),
		CPULimitMilli:   allLimits.Cpu().ScaledValue(resource.Milli),
		MemoryLimitMega: allLimits.Memory().ScaledValue(resource.Mega),

		Nodes: Nodes{
			NodesCPUIdleMilli:   nodesCPUIdleMilli,
			NodesMemoryFreeMega: nodesMemoryFreeMega,
		},
	}
	return res, nil
}

// elastic job filter.
func elastic(j *padv1.TrainingJob) bool {
	return j.Elastic()
}

// 
func isRunning(j *padv1.TrainingJob) bool {
	if j.Status.Phase == padv1.TrainingJobPhaseRunning || j.Status.Phase == padv1.TrainingJobPhaseScaling {
		return true
	} else {
		return false
	}
}


// sortedJobs return the names of sorted jobs by fulfillment and
// tiebreakers in ascending order.
func sortedJobs(j *sync.Map, filters ...func(*padv1.TrainingJob) bool) []*padv1.TrainingJob {
	var js trainingjobSlice
	for _, f := range filters {
		j.Range(func(k, v interface{}) bool {
			up := v.(*updater.TrainingJobUpdater)
			if !f(up.Job) {
				return true
			}
			js = append(js, up.Job)
			return true
		})
	}
	sort.Sort(js)
	return js
}

func searchAssignableNode(r *ClusterResource, j *padv1.TrainingJob) string {
	for name, idle := range r.Nodes.NodesCPUIdleMilli {
		if j.TrainerCPURequestMilli() <= idle &&
			j.TrainerMemRequestMega() <= r.Nodes.NodesMemoryFreeMega[name] {
			return name
		}
	}
	return ""
}

func scaleDryRun(r *ClusterResource, j *padv1.TrainingJob, curDiff int32, maxLoadDesired float64, scaleDown bool)(additional int) {
	additionalGPUInstance := 0
	additionalCPUInstance := 0
	cpuRequestMilli := j.TrainerCPURequestMilli()
	memRequestMega := j.TrainerMemRequestMega()
	gpuLimit := j.TrainerGPULimit()
	nodeName := ""
	// Adjust resource upon return.
	defer func() {
		r.GPULimit += gpuLimit * additional
		r.CPURequestMilli += cpuRequestMilli * int64(additional)
		r.MemoryRequestMega += memRequestMega * int64(additional)
		if nodeName != "" {
			r.Nodes.NodesCPUIdleMilli[nodeName] += cpuRequestMilli * int64(additional)
			r.Nodes.NodesMemoryFreeMega[nodeName] += memRequestMega * int64(additional)
		}
	}()

	// TODO(helin): j.TrainerJob.Spec.Parallelism may not reflect
	// the actual pod running for the trainer job. We need to
	// count the pod manually. And calculate the additional value
	// based on the running pod count,
	// j.TrainerJob.Spec.Parallelism, and curDiff.
	plannedInstance := int(*j.Spec.Trainer.ReplicaSpec.Spec.Parallelism) + int(curDiff)
	instanceMax := j.Spec.Trainer.MaxInstance
	instanceMin := j.Spec.Trainer.MinInstance

	// TODO(typhoonzero): refine below code to remove direction
	// ======================= scaleDown ======================
	if scaleDown {
		if plannedInstance > instanceMax {
			additional = -1
			return
		}
		gpuCondition := r.GPULimit > int(float64(r.GPUTotal)*maxLoadDesired)
		cpuCondition := r.CPURequestMilli > int64(float64(r.CPUTotalMilli)*maxLoadDesired)
		if gpuCondition || cpuCondition {
			if plannedInstance > instanceMin {
				additional = -1
				return
			}

			// can not scale down further
			additional = 0
			return
		}
		// do not try to scale up
		return
	}
	// ======================= scaleUp ==========================

	if plannedInstance >= instanceMax {
		// Do not scale or scale down, don't need to check if
		// there are available free resources.
		additional = instanceMax - plannedInstance
		return
	}

	if r.MemoryTotalMega-r.MemoryRequestMega <= memRequestMega {
		// insufficient memory, do not scale
		additional = 0
		return
	}
	if nodeName = searchAssignableNode(r, j); nodeName == "" {
		additional = 0
		return
	}

	// NOTE: do not scale up to use full cluster resource of CPU
	//       but we do scale up for GPU.
	if int64(float64(r.CPUTotalMilli)*maxLoadDesired)-r.CPURequestMilli >= cpuRequestMilli {
		additionalCPUInstance = 1
	}

	needGPU := gpuLimit > 0
	if needGPU && r.GPUTotal-r.GPULimit >= gpuLimit {
		additionalGPUInstance = 1
	}

	if needGPU {
		if additionalGPUInstance < additionalCPUInstance {
			additional = additionalGPUInstance
		} else {
			additional = additionalCPUInstance
		}
	} else {
		additional = additionalCPUInstance
	}

	return
}

func (a *Autoscaler) setAdditional(diff map[string]int32) {
	a.Jobupdater.Range(func(k, v interface{}) bool {
		key := k.(string)
		up := v.(*updater.TrainingJobUpdater)
		_, ok := diff[key]
		if !ok {
			up.Additional = diff[key]
		} else {
			up.Additional =  0
		}
		a.Jobupdater.Store(k, up)
		return true
	})
}

// scaleAllJobsDryRun pretends to rescale all jobs in order to find
// out the number of pods should be added/deleted for each job, or
// say, delta.  It returns a map from job name to the delta.
func (a *Autoscaler) scaleAllJobsDryRun(r ClusterResource, maxLoadDesired float64) {
	// Iteratively calculate scaling diff until nothing changes.
	diff := make(map[string]int32)
	for {
		noChange := true
		sorted := sortedJobs(a.Jobupdater, elastic, isRunning)
		dryRun := func(j *padv1.TrainingJob, isScaleDown bool) {
			name := j.Namespace + "/" +j.Name
			additional := scaleDryRun(&r, j, diff[name], maxLoadDesired, isScaleDown)
			diff[name] += int32(additional)

			if additional != 0 {
				noChange = false
			}
		}

		// TODO(typhoonzero): implement GPU priority CFS scheduler from here.

		// scale up from the ones that need scaling up the
		// most.
		for _, j := range sorted {
			dryRun(j, false)
		}

		// scale down from the ones that need scaling up the
		// least.
		for i := len(sorted) - 1; i >= 0; i-- {
			dryRun(sorted[i], true)
		}

		if noChange {
			break
		}
	}

	a.setAdditional(diff)
}

func (a *Autoscaler) scaleAllJobs() {
	a.Jobupdater.Range(func(k, v interface{}) bool {
		up := v.(*updater.TrainingJobUpdater)
		if up.Additional != int32(0) {
			log.Infof("additional of trainingjob %v not equal 0, scale it", k)
			up.Scale()
		}
		return true
	})
}

// Run monitors the cluster resources and training jobs in a loop,
// scales the training jobs according to the cluster resource.
func (a *Autoscaler) Run() {
	ticker := time.NewTicker(defaultLoopDur)
	defer ticker.Stop()
	log.Infof("start Autoscaler")
	for {
		<-ticker.C
		r, err := a.InquiryResource()
		if err != nil {
			log.Errorf("InquiryResource error=%v", err.Error())
			continue
		}
		log.Infof("Cluster.InquiryResource done", "resource", r)

		a.scaleAllJobsDryRun(r, a.MaxLoadDesired)
		a.scaleAllJobs()
	}
}
