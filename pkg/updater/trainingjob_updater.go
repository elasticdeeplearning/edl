package updater

import (
	"errors"
	"fmt"
	"reflect"

	log "github.com/inconshreveable/log15"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	typedcorev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"

	paddlev1 "github.com/paddlepaddle/edl/pkg/apis/paddlepaddle/v1"
	trainingJobClient "github.com/paddlepaddle/edl/pkg/client/clientset/versioned"
	"github.com/paddlepaddle/edl/pkg/client/clientset/versioned/scheme"
)

var (
	// ErrorUnkownResourceType not supported resource
	ErrorUnkownResourceType = errors.New("UnknownResourceType")
)

// JobUpdater controls the life circle of one TrainingJob instance
type JobUpdater struct {
	Job            *paddlev1.TrainingJob
	kubeCli        kubernetes.Interface
	trainingJobCli trainingJobClient.Interface
	status         paddlev1.TrainingJobStatus
	recorder       record.EventRecorder
	autoclean      bool
	Additional     int32
}

// NewJobUpdater returns JobUpdater instance
func NewJobUpdater(job *paddlev1.TrainingJob, kubeCli kubernetes.Interface, jobCli trainingJobClient.Interface, auto bool) *JobUpdater {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(log.Info)
	eventBroadcaster.StartRecordingToSink(&typedcorev1.EventSinkImpl{Interface: kubeCli.CoreV1().Events("")})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: "TrainingJobController"})

	return &JobUpdater{
		Job:            job,
		kubeCli:        kubeCli,
		trainingJobCli: jobCli,
		status:         *job.Status.DeepCopy(),
		recorder:       recorder,
		autoclean:      auto,
	}
}

// UID return uid of a job instance
func (j *JobUpdater) UID() types.UID {
	return j.Job.ObjectMeta.UID
}

// Update updates jobupdater's job instance
func (j *JobUpdater) Update(job *paddlev1.TrainingJob) {
	log.Debug("Updating", "job", j.FullName(), "statue", job.Status)
	j.Job = job
}

// GetJob returns trainingjob instance
func (j *JobUpdater) GetJob() *paddlev1.TrainingJob {
	return j.Job
}

// Delete deletes trainingjob instance
func (j *JobUpdater) Delete() error {
	return j.deleteTrainingJob()
}

// FullName returns job's namespace and name
func (j *JobUpdater) FullName() string {
	return fmt.Sprintf("%s/%s", j.Job.Namespace, j.Job.Name)
}

func (j *JobUpdater) masterName() string {
	return fmt.Sprintf("%s/%s", j.Job.Namespace, j.Job.Spec.Master.ReplicaSpec.Name)
}

func (j *JobUpdater) pserverName() string {
	return fmt.Sprintf("%s/%s", j.Job.Namespace, j.Job.Spec.Pserver.ReplicaSpec.Name)
}

func (j *JobUpdater) trainerName() string {
	return fmt.Sprintf("%s/%s", j.Job.Namespace, j.Job.Spec.Trainer.ReplicaSpec.Name)
}

// Reconcile tries to get the job into the desired state
func (j *JobUpdater) Reconcile() error {
	log.Info("Reconciling TrainingJob", "job", j.FullName(), "current status phase", j.Job.Status.Phase)

	if j.Job.ObjectMeta.DeletionTimestamp != nil {
		log.Info("Deleted timestamp", "job", j.FullName(), "timestamp", j.Job.ObjectMeta.DeletionTimestamp.String())
		return nil
	}

	if j.Job.Status.Phase == paddlev1.TrainingJobPhaseNone {
		log.Info("Setting up", "job", j.FullName())
		if err := j.setup(); err != nil {
			j.status.Phase = paddlev1.ResourceStateFailed
			j.status.Reason = err.Error()
			log.Error("Error setting up TrainingJob", "job", j.FullName(), "err", err.Error())
		} else {
			j.status.Phase = paddlev1.TrainingJobPhaseCreating
			log.Info("Finish setting up TrainingJob", "job", j.FullName())
		}
		if err := j.updateCRDStatus(); err != nil {
			log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}
	}

	if j.Job.Status.Phase == paddlev1.TrainingJobPhaseCreating {
		log.Info("Creating TrainingJob", "job", j.FullName())
		if err := j.createTrainingJob(); err != nil {
			log.Error("Error creating TrainingJob", "job", j.FullName(), "err", err.Error())
			j.status.Phase = paddlev1.ResourceStateFailed
			j.status.Reason = err.Error()
		} else {
			log.Info("Finish creating TrainingJob", "job", j.FullName())
		}

		if err := j.updateCRDStatus(); err != nil {
			log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}

		phase, reason, err := j.GetStatus()
		log.Info("Error creating TrainingJob", "job", j.FullName(), "current phase", phase, "reason", reason, "err", err)
		if err != nil {
			log.Error("Error get TrainingJob status", "job", j.FullName(), "err", err.Error())
			return err
		}

		j.status.Phase = phase
		j.status.Reason = reason

		if err := j.updateCRDStatus(); err != nil {
			log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}
	}

	if j.Job.Status.Phase == paddlev1.TrainingJobPhaseRunning {
		phase, reason, err := j.GetStatus()
		if err != nil {
			log.Error("Error get TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}

		j.status.Phase = phase
		j.status.Reason = reason
		if err := j.updateCRDStatus(); err != nil {
			log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}
	}

	if j.Job.Status.Phase == paddlev1.TrainingJobPhaseScaling {
		if j.Additional != 0 {
			if err := j.scale(); err != nil {
				//TODO
				return err
			}
			j.Additional = 0
		}

		phase, reason, err := j.GetStatus()
		if err != nil {
			log.Error("Error get TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}

		j.status.Phase = phase
		j.status.Reason = reason
		if err := j.updateCRDStatus(); err != nil {
			log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
			return err
		}
	}

	if j.Job.Status.Phase == paddlev1.TrainingJobPhaseSucceeded ||
		j.Job.Status.Phase == paddlev1.TrainingJobPhaseFailed {
		if j.autoclean {
			log.Info("Releasing TrainingJob resource", "job", j.FullName(), "current status phase", j.Job.Status.Phase)
			if err := j.releaseTrainer(); err != nil {
				log.Error("Error releasing TrainingJob trainer resource", "job", j.FullName(), "err", err.Error())
				return err
			}
			log.Info("Finish releasing TrainingJob trainer resource", "job", j.FullName())

			if err := j.releaseMasterRoles(); err != nil {
				log.Error("Error releasing TrainingJob master/pserver resource", "job", j.FullName(), "err", err.Error())
				return err
			}
			log.Info("Finish releasing TrainingJob master/pserver resource", "job", j.FullName())

			j.recorder.Event(j.Job, corev1.EventTypeNormal, "Terminated", "All pods cleaned")
		} else {
			j.recorder.Event(j.Job, corev1.EventTypeNormal, "Terminated", "All pods kept")
		}
	}

	if err := j.updateCRDStatus(); err != nil {
		log.Error("Error updating TrainingJob", "job", j.FullName(), "err", err.Error())
		return err
	}

	return nil
}

func (j *JobUpdater) setup() error {
	var parser DefaultJobParser
	var err error
	j.Job, err = parser.NewTrainingJob(j.Job)
	if err != nil {
		log.Error("error settting up", "job", j.FullName(), "err", err.Error())
	}

	return err
}

func (j *JobUpdater) updateCRDStatus() error {
	log.Debug("updating TrainingJob status", "job", j.FullName(), "former status", j.Job.Status, "current status", j.status)
	if reflect.DeepEqual(j.status, j.Job.Status) {
		log.Debug("update TrainingJob skipped", "job", j.FullName(), "status", j.status)
		return nil
	}

	newJob := j.Job
	newJob.Status = j.status
	// sync trainingjob to apiserver
	newJob, err := j.trainingJobCli.PaddlepaddleV1().TrainingJobs(j.Job.Namespace).Update(newJob)
	if err != nil {
		return err
	}

	j.Job = newJob
	return nil
}

// GetStatus get current status phase and reasion of job
func (j *JobUpdater) GetStatus() (paddlev1.TrainingJobPhase, string, error) {
	phase := j.status.Phase
	reason := ""

	trainers, err := j.kubeCli.BatchV1().Jobs(j.Job.Namespace).Get(j.Job.Spec.Trainer.ReplicaSpec.Name, v1.GetOptions{})
	if err != nil {
		log.Error("error getting trainers", "name", j.trainerName(), "err", err.Error())
		return phase, reason, err
	}

	// total running
	totalRunning, err := j.jobTotalRunning()
	if err != nil {
		return phase, reason, err
	} else if totalRunning {
		phase = paddlev1.TrainingJobPhaseRunning
		reason = "all pods are running"
	}

	// the parallelism of batch/job trainer will be modified after success/failure
	total := *j.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism
	if j.Job.Spec.FaultTolerant {
		if trainers.Status.Failed == total {
			phase = paddlev1.TrainingJobPhaseFailed
			reason = "all trainer instances have failed"
			return phase, reason, nil
		} else if trainers.Status.Succeeded == total && trainers.Status.Active == 0 {
			phase = paddlev1.TrainingJobPhaseSucceeded
			reason = "all trainer instances have done"
			return phase, reason, nil
		}
	} else {
		if trainers.Status.Failed != 0 {
			failedPods, err := j.findFailedTrainerPods()
			if err != nil {
				return phase, reason, err
			}

			podNameList := make([]string, 0)
			podNodeList := make([]string, 0)
			podReasonList := make([]string, 0)
			for _, pod := range failedPods {
				podNameList = append(podNameList, pod.Name)
				podNodeList = append(podNodeList, pod.Status.HostIP)
				podReasonList = append(podReasonList, pod.Status.Reason)
			}

			phase = paddlev1.TrainingJobPhaseFailed
			reason = fmt.Sprintf("trainer instances %s on %s have failed", podNameList, podNodeList)
			podFailReason := fmt.Sprintf("trainer instances %s on %s have failed, detailed reasons: %s", podNameList,
				podNodeList, podReasonList)
			j.recorder.Event(j.Job, corev1.EventTypeWarning, "Pods Failed", podFailReason)
			return phase, reason, nil
		} else if trainers.Status.Succeeded == total && trainers.Status.Active == 0 {
			phase = paddlev1.TrainingJobPhaseSucceeded
			reason = "all trainer instances have done"
			return phase, reason, nil
		}
	}

	if j.Additional != 0 {
		phase = paddlev1.TrainingJobPhaseScaling
		reason = fmt.Sprintf("need scale")
	}

	return phase, reason, nil
}

func (j *JobUpdater) createTrainingJob() error {
	if j.Job.Spec.FaultTolerant {
		log.Debug("creating master", "name", j.masterName())
		if err := j.createResource(paddlev1.MASTER); err != nil {
			return err
		}
	}

	log.Debug("creatint pserver", "name", j.pserverName())
	if err := j.createResource(paddlev1.PSERVER); err != nil {
		return err
	}

	log.Debug("creating trainer", "name", j.trainerName())
	if err := j.createTrainer(); err != nil {
		return err
	}

	return nil
}

func (j *JobUpdater) createResource(rt paddlev1.TrainingResourceType) error {
	resource := new(v1beta1.ReplicaSet)
	switch rt {
	case paddlev1.MASTER:
		resource = j.Job.Spec.Master.ReplicaSpec
	case paddlev1.PSERVER:
		resource = j.Job.Spec.Pserver.ReplicaSpec
	default:
		return ErrorUnkownResourceType
	}

	if _, err := j.kubeCli.ExtensionsV1beta1().ReplicaSets(resource.Namespace).Get(resource.Name, v1.GetOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			if _, err := j.kubeCli.ExtensionsV1beta1().ReplicaSets(resource.Namespace).Create(resource); err != nil {
				log.Error("error creating resource", "namespace", resource.Namespace, "name", resource.Name, "err", err.Error())
				return err
			}
			log.Debug("finish creating resource", "namespace", resource.Namespace, "name", resource.Name)
			return nil
		}
		log.Error("error getting resource", "namespace", resource.Namespace, "name", resource.Name, "err", err.Error())
		return err
	}

	log.Debug("resource already existing, skipping", "namespace", resource.Namespace, "name", resource.Name)
	return nil
}

func (j *JobUpdater) createTrainer() error {
	if _, err := j.kubeCli.BatchV1().Jobs(j.Job.Namespace).Get(j.Job.Spec.Trainer.ReplicaSpec.Name, v1.GetOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			if _, err = j.kubeCli.BatchV1().Jobs(j.Job.Namespace).Create(j.Job.Spec.Trainer.ReplicaSpec); err != nil {
				log.Error("error creating trainer", "name", j.trainerName(), "err", err.Error())
				return err
			}
			log.Debug("finishing creating trainer", "name", j.trainerName())
			return nil
		}
		log.Error("error getting trainer", "name", j.trainerName(), "err", err.Error())
		return err
	}

	log.Debug("trainer already existing skipping", "name", j.trainerName())
	return nil
}

func (j *JobUpdater) deleteTrainingJob() error {
	if j.Job.Spec.FaultTolerant {
		log.Debug("deleting master", "name", j.masterName())
		if err := j.deleteResource(paddlev1.MASTER); err != nil {
			log.Error("error deleting master", "name", j.masterName(), "err", err.Error())
			return err
		}
	}

	log.Debug("deleting pserver", "name", j.pserverName())
	if err := j.deleteResource(paddlev1.PSERVER); err != nil {
		log.Error("error deleting: %s, err: %s", j.pserverName(), err.Error())
		return err
	}

	log.Debug("deleting trainer", "name", j.trainerName())
	if err := j.deleteTrainer(); err != nil {
		log.Error("error deleting trainer", "name", j.trainerName(), "err", err.Error())
		return err
	}

	return nil
}

func (j *JobUpdater) deleteResource(rt paddlev1.TrainingResourceType) error {
	if err := j.releaseResource(rt); err != nil {
		return err
	}

	resourceName := j.Job.Name + "-" + string(rt)
	if err := j.kubeCli.ExtensionsV1beta1().ReplicaSets(j.Job.Namespace).Delete(resourceName, &v1.DeleteOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			log.Debug("resource not found, skipped", "namespace", j.Job.Namespace, "name", resourceName)
			return nil
		}
		return err
	}
	log.Debug("finishing releasing", "namespace", j.Job.Namespace, "name", resourceName)
	return nil
}

func (j *JobUpdater) deleteTrainer() error {
	if err := j.releaseTrainer(); err != nil {
		return err
	}

	if err := j.kubeCli.BatchV1().Jobs(j.Job.Namespace).Delete(j.Job.Spec.Trainer.ReplicaSpec.Name, &v1.DeleteOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			log.Debug("trainer not exist skipped", "name", j.trainerName())
			return nil
		}
		return err
	}
	log.Debug("finishing deleting trainer", "name", j.trainerName())
	return nil
}

func (j *JobUpdater) releaseMasterRoles() error {
	if j.Job.Spec.FaultTolerant {
		if err := j.releaseResource(paddlev1.MASTER); err != nil {
			log.Error("error releasing master", "name", j.masterName(), "err", err)
			return err
		}
	}

	if err := j.releaseResource(paddlev1.PSERVER); err != nil {
		log.Error("error releasing pserver", "name", j.pserverName(), "err", err)
		return err
	}

	return nil
}

func (j *JobUpdater) releaseResource(rt paddlev1.TrainingResourceType) error {
	resourceName := ""
	switch rt {
	case paddlev1.MASTER:
		resourceName = j.Job.Spec.Master.ReplicaSpec.Name
	case paddlev1.PSERVER:
		resourceName = j.Job.Spec.Pserver.ReplicaSpec.Name
	default:
		return ErrorUnkownResourceType
	}

	resource, getErr := j.kubeCli.ExtensionsV1beta1().ReplicaSets(j.Job.Namespace).Get(resourceName, v1.GetOptions{})
	if getErr != nil {
		if apierrors.IsNotFound(getErr) {
			log.Debug("resouce instance not exist, skipped", "namespace", j.Job.Namespace, "name", resourceName)
			return nil
		}
		log.Error("error getting instance", "namespace", j.Job.Namespace, "name", resourceName, "err", getErr)
		return getErr
	}

	if *resource.Spec.Replicas != 0 {
		var replicas int32
		replicas = 0
		resource.Spec.Replicas = &replicas
		if _, err := j.kubeCli.ExtensionsV1beta1().ReplicaSets(j.Job.Namespace).Update(resource); err != nil {
			log.Error("error setting replicas to 0", "namespace", j.Job.Namespace, "name", resourceName, "err", err.Error())
			return err
		}
	}

	if resource.Status.FullyLabeledReplicas != 0 {
		key := "paddle-job-" + rt
		labels := Labels(map[string]string{
			string(key): j.Job.Name,
		})

		selector, _ := labels.LabelsParser()
		options := v1.ListOptions{
			LabelSelector: selector,
		}

		if err := j.kubeCli.CoreV1().Pods(j.Job.Namespace).DeleteCollection(&v1.DeleteOptions{}, options); err != nil {
			log.Error("error deleting resource pods", "namespace", j.Job.Namespace, "name", resourceName, "err", err.Error())
			return err
		}
	}

	return nil
}

func (j *JobUpdater) releaseTrainer() error {
	jobNs := j.Job.Namespace
	jobName := j.Job.Spec.Trainer.ReplicaSpec.Name

	jobSpec, getErr := j.kubeCli.BatchV1().Jobs(jobNs).Get(jobName, v1.GetOptions{})
	if getErr != nil {
		if apierrors.IsNotFound(getErr) {
			return nil
		}
		log.Error("error getting job spec for TrainingJob trainer", "name", j.trainerName())
		return getErr
	}

	if *jobSpec.Spec.Parallelism != 0 {
		log.Debug("reset parallelism to zero for TrainingJob trainer", "name", j.trainerName())
		var parallism int32
		parallism = 0
		jobSpec.Spec.Parallelism = &parallism
		if _, err := j.kubeCli.BatchV1().Jobs(jobNs).Update(jobSpec); err != nil {
			log.Error("error resetting parallelism for TrainingJob trainer", "name", j.trainerName())
			return err
		}
	}

	labels := Labels(map[string]string{
		"paddle-job": j.Job.Name,
	})
	selector, _ := labels.LabelsParser()
	options := v1.ListOptions{
		LabelSelector: selector,
	}

	if err := j.kubeCli.CoreV1().Pods(jobNs).DeleteCollection(&v1.DeleteOptions{}, options); err != nil {
		log.Error("error deleting pods of TrainingJob trainer", "name", j.trainerName())
		return err
	}

	return nil
}

func (j *JobUpdater) jobTotalRunning() (bool, error) {
	if j.Job.Spec.FaultTolerant {
		masterRunning, err := j.masterRoleTotalRunning(paddlev1.MASTER)
		if err != nil {
			return false, err
		}
		if !masterRunning {
			return false, nil
		}
	}

	pserverRunning, err := j.masterRoleTotalRunning(paddlev1.PSERVER)
	if err != nil {
		return false, err
	}
	if !pserverRunning {
		return false, nil
	}

	return j.trainerTotalRunning()
}

func (j *JobUpdater) masterRoleTotalRunning(rt paddlev1.TrainingResourceType) (bool, error) {
	var resourceName string
	switch rt {
	case paddlev1.MASTER:
		resourceName = j.Job.Spec.Master.ReplicaSpec.Name
	case paddlev1.PSERVER:
		resourceName = j.Job.Spec.Pserver.ReplicaSpec.Name
	default:
		return false, ErrorUnkownResourceType
	}
	resource, err := j.kubeCli.ExtensionsV1beta1().ReplicaSets(j.Job.Namespace).Get(resourceName, v1.GetOptions{})
	if err != nil {
		return false, err
	}

	log.Debug("resource status", "namespace", j.Job.Namespace, "name", resourceName, "status", resource.Status)
	if resource.Status.ReadyReplicas >= *resource.Spec.Replicas {
		return true, nil
	}
	return false, nil
}

func (j *JobUpdater) trainerTotalRunning() (bool, error) {
	trainerName := j.Job.Spec.Trainer.ReplicaSpec.Name
	trainers, err := j.kubeCli.BatchV1().Jobs(j.Job.Namespace).Get(trainerName, v1.GetOptions{})
	if err != nil {
		return false, err
	}

	log.Debug("trainer status", "namespace", j.Job.Namespace, "name", trainerName, "status", trainers.Status)
	podsList, err := j.kubeCli.CoreV1().Pods(j.Job.Namespace).List(v1.ListOptions{LabelSelector: "paddle-job=" + j.Job.Name})
	var runningPodCount int32
	for _, pod := range podsList.Items {
		if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodSucceeded {
			runningPodCount++
		}
	}

	if runningPodCount == *trainers.Spec.Parallelism {
		return true, nil
	}
	return false, nil
}

func (j *JobUpdater) findFailedTrainerPods() ([]*corev1.Pod, error) {
	failedPods := make([]*corev1.Pod, 0)

	podsList, err := j.kubeCli.CoreV1().Pods(j.Job.Namespace).List(v1.ListOptions{LabelSelector: "paddle-job=" + j.Job.Name})
	if err != nil {
		return failedPods, err
	}
	for _, pod := range podsList.Items {
		if pod.Status.Phase == corev1.PodFailed {
			failedPods = append(failedPods, &pod)
		}
	}

	return failedPods, nil
}

func (j *JobUpdater) scale() (err error) {
	jobNs := j.Job.Namespace
	jobName := j.Job.Spec.Trainer.ReplicaSpec.Name
	jobSpec, err := j.kubeCli.BatchV1().Jobs(jobNs).Get(jobName, v1.GetOptions{})
	if err != nil {
		return err
	}

	newParallelism := *jobSpec.Spec.Parallelism + j.Additional
	newBackoffLimit := *jobSpec.Spec.BackoffLimit
	if j.Additional < 0 {
		newBackoffLimit -= j.Additional
	}
	jobSpec.Spec.Parallelism = &newParallelism
	jobSpec.Spec.BackoffLimit = &newBackoffLimit
	j.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism = &newParallelism
	log.Debug("scaling job", "namespace", jobNs, "name", jobName, "new instance num", newParallelism)
	if _, err := j.kubeCli.BatchV1().Jobs(jobNs).Update(jobSpec); err != nil {
		log.Debug("failed to scale job", "namespace", jobNs, "name", jobName, "error", err.Error())
		return err
	}

	return nil
}
