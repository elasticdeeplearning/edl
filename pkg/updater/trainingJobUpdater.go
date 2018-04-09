package updater

import (
	"fmt"
	log "github.com/golang/glog"

	padv1 "github.com/paddlepaddle/edl/pkg/apis/paddlepaddle/v1"
	trainingJobClient "github.com/paddlepaddle/edl/pkg/client/clientset/versioned"

	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"reflect"
	"time"
)

const (
	retry                 = 5
	retryTime             = 5 * time.Second
	convertedTimerTicker  = 10 * time.Second
	confirmResourceTicker = 5 * time.Second
	eventChLength         = 1000
	factor                = 0.8
)

type trainingJobEventType string

const (
	trainingJobEventDelete trainingJobEventType = "Delete"
	trainingJobEventModify trainingJobEventType = "Modify"
	trainingJobEventScale  trainingJobEventType = "Scale"
)

type trainingJobEvent struct {
	// pet is the TrainingJobEventType of TrainingJob
	pet trainingJobEventType
	// The job transfer the information fo job
	job *padv1.TrainingJob
	// additional is the num to scale
	additional int32
}

// TrainingJobUpdater is used to manage a specific TrainingJob
type TrainingJobUpdater struct {
	// Job is the job the TrainingJob manager.
	Job *padv1.TrainingJob

	// kubeClient is standard kubernetes client.
	KubeClient kubernetes.Interface

	// TrainingJobClient is the client of TrainingJob.
	TrainingJobClient trainingJobClient.Interface

	// Status is the status in memory, update when TrainingJob status changed and update the CRD resource status.
	Status padv1.TrainingJobStatus

	// EventCh receives events from the controller, include Modify and Delete.
	// When trainingJobEvent is Delete it will delete all resources
	// The capacity is 1000.
	EventCh chan *trainingJobEvent

	// Additional is the num scale.
	Additional int32
}

// NewUpdater creates a new TrainingJobUpdater and start a goroutine to control current job.
func NewUpdater(job *padv1.TrainingJob, kubeClient kubernetes.Interface, trainingJobClient trainingJobClient.Interface) (*TrainingJobUpdater,
	error) {
	log.Infof("NewJobber namespace=%v name=%v", job.Namespace, job.Name)
	updater := &TrainingJobUpdater{
		Job:               job,
		KubeClient:        kubeClient,
		TrainingJobClient: trainingJobClient,
		Status:            job.Status,
		EventCh:           make(chan *trainingJobEvent, eventChLength),
	}
	go updater.start()
	return updater, nil
}

// Notify is used to receive event from controller. While controller receive a informer,
// it will notify updater to process the event. It send event to updater's eventCh.
func (updater *TrainingJobUpdater) notify(te *trainingJobEvent) {
	updater.EventCh <- te
	lene, cape := len(updater.EventCh), cap(updater.EventCh)
	if lene > int(float64(cape)*factor) {
		log.Warning("the len of updater eventCh ", updater.Job.Name, " is near to full")
	}
}

// Delete send a delete event to updater, updater will kill the trainingjob and clear all the resource of the
// trainingjob.
func (updater *TrainingJobUpdater) Delete() {
	updater.notify(&trainingJobEvent{pet: trainingJobEventDelete})
}

// Modify send a modify event to updater, updater will processing according to the situation.
func (updater *TrainingJobUpdater) Modify(nj *padv1.TrainingJob) {
	updater.notify(&trainingJobEvent{pet: trainingJobEventModify, job: nj})
}

// Scale send a scale event to updater, updater will scale the job to desire replicas.
func (updater *TrainingJobUpdater) Scale() {
	updater.notify(&trainingJobEvent{pet: trainingJobEventScale})
}

func (updater *TrainingJobUpdater) releaseResource(tp padv1.TrainingResourceType) error {
	resource := new(v1beta1.ReplicaSet)
	switch tp {
	case padv1.Master:
		resource = updater.Job.Spec.Master.ReplicaSpec
	case padv1.Pserver:
		resource = updater.Job.Spec.Pserver.ReplicaSpec
	default:
		return fmt.Errorf("unknow resource")
	}
	var replica int32
	resource.Spec.Replicas = &replica
	_, err := updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Update(resource)
	if errors.IsNotFound(err) {
		return err
	}
	key := "paddle-job-" + tp

	labels := Labels(map[string]string{
		string(key): updater.Job.Name,
	})

	selector, _ := labels.LabelsParser()
	options := metav1.ListOptions{
		LabelSelector: selector,
	}

	for j := 0; j <= retry; j++ {
		time.Sleep(confirmResourceTicker)
		pl, err := updater.KubeClient.CoreV1().Pods(updater.Job.Namespace).List(options)
		if err == nil && len(pl.Items) == 0 {
			return nil
		}
	}
	return updater.KubeClient.CoreV1().Pods(updater.Job.Namespace).DeleteCollection(&metav1.DeleteOptions{}, options)
}

func (updater *TrainingJobUpdater) releaseMaster() error {
	return updater.releaseResource(padv1.Master)
}

func (updater *TrainingJobUpdater) releasePserver() error {
	return updater.releaseResource(padv1.Pserver)
}

func (updater *TrainingJobUpdater) releaseTrainer() error {
	labels := Labels(map[string]string{
		"paddle-job": updater.Job.Name,
	})
	selector, _ := labels.LabelsParser()
	options := metav1.ListOptions{
		LabelSelector: selector,
	}

	return updater.KubeClient.CoreV1().Pods(updater.Job.Namespace).DeleteCollection(&metav1.DeleteOptions{}, options)
}

func (updater *TrainingJobUpdater) deleteTrainingJob() error {
	fault := false
	log.Infof("Start to delete TrainingJob namespace=%v name=%v", updater.Job.Namespace, updater.Job.Name)
	if updater.Job.Spec.FaultTolerant {
		log.Infof("Release master, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
		if err := updater.releaseMaster(); err != nil {
			log.Error(err.Error())
			fault = true
		}
	}

	log.Infof("Release pserver, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
	if err := updater.releasePserver(); err != nil {
		log.Error(err.Error())
		fault = true
	}

	if updater.Job.Spec.FaultTolerant {
		log.Infof("Deleting TrainingJob matadata, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Master.ReplicaSpec.Name)
		if err := updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Delete(updater.Job.Spec.Master.ReplicaSpec.Name, &metav1.DeleteOptions{}); err != nil {
			log.Error("delete master replicaset error: ", err.Error())
			fault = true
		}
	}

	log.Infof("Deleting TrainingJob matadata, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Pserver.ReplicaSpec.Name)
	if err := updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Delete(updater.Job.Spec.Pserver.ReplicaSpec.Name, &metav1.DeleteOptions{}); err != nil {
		log.Error("delete pserver replicaset error: ", err.Error())
		fault = true
	}

	log.Infof("Deleting TrainingJob matadata, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
	if err := updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).Delete(updater.Job.Spec.Trainer.ReplicaSpec.Name, &metav1.DeleteOptions{}); err != nil {
		log.Error("delete trainer replicaset error: ", err.Error())
		fault = true
	}

	log.Infof("Release trainer, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
	if err := updater.releaseTrainer(); err != nil {
		log.Error("release trainer  error: ", err.Error())
		fault = true
	}

	log.Infof("End to delete TrainingJob namespace=%v name=%v", updater.Job.Namespace, updater.Job.Name)

	if fault {
		return fmt.Errorf("delete resource error namespace=%v name=%v", updater.Job.Namespace, updater.Job.Name)
	}
	return nil
}

func (updater *TrainingJobUpdater) createResource(tp padv1.TrainingResourceType) error {
	resource := new(v1beta1.ReplicaSet)
	switch tp {
	case padv1.Master:
		resource = updater.Job.Spec.Master.ReplicaSpec
	case padv1.Pserver:
		resource = updater.Job.Spec.Pserver.ReplicaSpec
	default:
		return fmt.Errorf("unknown resource")
	}
	for {
		_, err := updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Get(resource.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			log.Infof("Not found to create namespace=%v name=%v resourceName=%v", updater.Job.Namespace, updater.Job.Name, resource.Name)
			_, err = updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Create(resource)
			if err != nil {
				updater.Status.Phase = padv1.TrainingJobPhaseFailed
				updater.Status.Reason = "Internal error; create resource error:" + err.Error()
				return err
			}
		} else if err != nil {
			log.Errorf("Get resource error, namespace=%v name=%v resourceName=%v error=%v", updater.Job.Namespace, updater.Job.Name, resource.Name, err.Error())
			time.Sleep(retryTime)
			continue
		}
		ticker := time.NewTicker(confirmResourceTicker)
		defer ticker.Stop()
		for v := range ticker.C {
			rs, err := updater.KubeClient.ExtensionsV1beta1().ReplicaSets(updater.Job.Namespace).Get(resource.Name, metav1.GetOptions{})
			log.Infof("Current time %v runing pod is %v, resourceName=%v", v.String(), rs.Status.ReadyReplicas, resource.Name)
			if err != nil && !errors.IsServerTimeout(err) && !errors.IsTooManyRequests(err) {
				updater.Status.Reason = "Internal error; create resource error:" + err.Error()
				return err
			}
			if errors.IsServerTimeout(err) || errors.IsTooManyRequests(err) {
				log.Warningf("Connect to kubernetes failed for reasons=%v, retry next ticker", err.Error())
				continue
			}
			if *resource.Spec.Replicas == 0 {
				return fmt.Errorf(" trainingjob is deleting, namespace=%v name=%v ", updater.Job.Namespace, updater.Job.Name)

			}
			if rs.Status.ReadyReplicas == *resource.Spec.Replicas {
				log.Infof("Create resource done , namespace=%v name=%v resourceName=%v", updater.Job.Namespace, updater.Job.Name, resource.Name)
				return nil
			}
		}
	}
}

func (updater *TrainingJobUpdater) createTrainer() error {
	resource := updater.Job.Spec.Trainer.ReplicaSpec
	for {
		_, err := updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).Get(resource.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			log.Infof("not found to create trainer namespace=%v name=%v", updater.Job.Namespace, updater.Job.Name)
			_, err = updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).Create(resource)
			if err != nil {
				updater.Status.Phase = padv1.TrainingJobPhaseFailed
				updater.Status.Reason = "Internal error; create trainer error:" + err.Error()
				return err
			}
		} else if err != nil {
			log.Errorf("Get resource error, namespace=%v name=%v resourceName=%v error=%v", updater.Job.Namespace, updater.Job.Name, resource.Name, err.Error())
			time.Sleep(retryTime)
			continue
		}
		updater.Status.Phase = padv1.TrainingJobPhaseRunning
		updater.Status.Reason = ""
		return nil
	}
}

func (updater *TrainingJobUpdater) createTrainingJob() error {
	if updater.Job.Spec.FaultTolerant {

		if err := updater.createResource(padv1.Master); err != nil {
			return err
		}
	}
	if err := updater.createResource(padv1.Pserver); err != nil {
		return err
	}
	return updater.createTrainer()
}

func (updater *TrainingJobUpdater) updateCRDStatus() error {
	if reflect.DeepEqual(updater.Status, updater.Job.Status) {
		return nil
	}
	newTrainingJob := updater.Job
	newTrainingJob.Status = updater.Status
	newTrainingJob, err := updater.TrainingJobClient.PaddlepaddleV1().TrainingJobs(updater.Job.Namespace).Update(newTrainingJob)
	if err != nil {
		return err
	}
	updater.Job = newTrainingJob
	return nil
}

// parseTrainingJob validates the fields and parses the TrainingJob
func (updater *TrainingJobUpdater) parseTrainingJob() {
	if updater.Job == nil {
		updater.Status.Phase = padv1.TrainingJobPhaseFailed
		updater.Status.Reason = "Internal error; Setup error; job is missing TainingJob"
		return
	}

	var parser DefaultJobParser
	var creatErr error
	updater.Job, creatErr = parser.NewTrainingJob(updater.Job)

	if creatErr != nil {
		updater.Status.Phase = padv1.TrainingJobPhaseFailed
		updater.Status.Reason = creatErr.Error()
	} else {
		updater.Status.Phase = padv1.TrainingJobPhaseCreating
		updater.Status.Reason = ""
	}
}

func (updater *TrainingJobUpdater) getTrainerReplicaStatuses() ([]*padv1.TrainingResourceStatus, error) {
	var replicaStatuses []*padv1.TrainingResourceStatus
	trs := padv1.TrainingResourceStatus{
		TrainingResourceType: padv1.Trainer,
		State:                padv1.ResourceStateNone,
		ResourceStates:       make(map[padv1.ResourceState]int),
	}
	// TODO(ZhengQi): get detail status in future
	replicaStatuses = append(replicaStatuses, &trs)
	return replicaStatuses, nil
}

// GetStatus get TrainingJob status from trainers.
func (updater *TrainingJobUpdater) GetStatus() (*padv1.TrainingJobStatus, error) {

	status := updater.Status

	j, err := updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).
		Get(updater.Job.Spec.Trainer.ReplicaSpec.Name, metav1.GetOptions{})
	if err != nil {
		log.Error("get trainer error:", err.Error())
		return &status, err
	}

	status.ReplicaStatuses, err = updater.getTrainerReplicaStatuses()
	if err != nil {
		log.Error("get trainer replica status error:", err.Error())
	}

	if updater.Job.Spec.FaultTolerant {
		// TODO(ZhengQi): should to confirm when job done
		if j.Status.Failed == *updater.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism {
			status.Phase = padv1.TrainingJobPhaseFailed
			status.Reason = "all trainer have failed!"
		} else {
			if j.Status.Succeeded != 0 && j.Status.Active == 0 {
				status.Phase = padv1.TrainingJobPhaseSucceeded
				status.Reason = "Success!"
			}
		}
	} else {
		if j.Status.Failed != 0 {
			status.Phase = padv1.TrainingJobPhaseFailed
			status.Reason = "at least one trainer failed!"
		} else {
			if j.Status.Succeeded == *updater.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism && j.Status.Active == 0 {
				status.Phase = padv1.TrainingJobPhaseSucceeded
				status.Reason = "all trainer have succeeded!"
			}
		}
	}
	return &status, nil
}

// Convert is main process to convert TrainingJob to desire status.
func (updater *TrainingJobUpdater) Convert() {
	log.Infof("convert status, namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)

	if updater.Status.Phase == padv1.TrainingJobPhaseRunning || updater.Status.Phase == padv1.TrainingJobPhaseScaling {
		status, err := updater.GetStatus()
		if err != nil {
			log.Error("get current status of trainer from k8s error:", err.Error())
			return
		}
		updater.Status = *status.DeepCopy()
		log.Infof("Current status namespace=%v name=%v status=%v : ", updater.Job.Namespace, updater.Job.Name, status)
		err = updater.updateCRDStatus()
		if err != nil {
			log.Warning("get current status to update trainingJob status error: ", err.Error())
		}
		if updater.Status.Phase == padv1.TrainingJobPhaseSucceeded || updater.Status.Phase == padv1.TrainingJobPhaseFailed {
			log.Infof("Release Resource namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
			if updater.Job.Spec.FaultTolerant {
				log.Infof("Release master, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
				if err := updater.releaseMaster(); err != nil {
					log.Error(err.Error())
				}
			}
			log.Infof("Release pserver, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
			if err := updater.releasePserver(); err != nil {
				log.Error(err.Error())
			}
		}
	}
}

// InitResource is used to parse trainingJob and create trainingJob resources.
func (updater *TrainingJobUpdater) InitResource() {
	if updater.Status.Phase == padv1.TrainingJobPhaseNone {
		log.Infof("set up trainingJob namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
		updater.parseTrainingJob()
		err := updater.updateCRDStatus()
		if err != nil {
			log.Warning("set up trainingJob to update trainingJob status error: ", err.Error())
		}
	}

	if updater.Status.Phase == padv1.TrainingJobPhaseCreating {
		log.Infof("create trainingJob namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
		_ = updater.createTrainingJob()
		err := updater.updateCRDStatus()
		if err != nil {
			log.Warning("create trainingJob to update trainingJob status error: ", err.Error())
		}
		if updater.Status.Phase == padv1.TrainingJobPhaseFailed {
			log.Infof("Release Resource for failed namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
			if updater.Job.Spec.FaultTolerant {
				log.Infof("Release master, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
				if err := updater.releaseMaster(); err != nil {
					log.Error(err.Error())
				}
			}

			log.Infof("Release pserver, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Spec.Trainer.ReplicaSpec.Name)
			if err := updater.releasePserver(); err != nil {
				log.Error(err.Error())
			}
		}
	}
}

func (updater *TrainingJobUpdater) scale(additional int32) *padv1.TrainerJobScaleRecord {

	scaleRecord := &padv1.TrainerJobScaleRecord{
		ScaleTimestamp: metav1.NewTime(time.Now()),
		Additional:     additional,
	}
	resource := updater.Job.Spec.Trainer.ReplicaSpec
	*resource.Spec.Parallelism = *resource.Spec.Parallelism + additional

	_, err := updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).Update(resource)
	if err != nil {
		scaleRecord.Status = padv1.ScaleFalse
		scaleRecord.Reason = err.Error()
	} else {
		updater.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism = resource.Spec.Parallelism
		scaleRecord.Status = padv1.ScaleTrue
		scaleRecord.Reason = ""
		updater.Status.Phase = padv1.TrainingJobPhaseScaling
	}
	updater.Status.ScaleRecords.ScaleRecords = append(updater.Status.ScaleRecords.ScaleRecords, scaleRecord)
	return scaleRecord
}

func (updater *TrainingJobUpdater) syncScale() {
	for {
		if updater.Status.Phase == padv1.TrainingJobPhaseSucceeded || updater.Status.Phase == padv1.TrainingJobPhaseFailed {
			log.Infof("Omit sync scale for job have done, namespace=%v name=%v", updater.Job.Namespace, updater.Job.Name)
			return
		}

		j, err := updater.KubeClient.BatchV1().Jobs(updater.Job.Namespace).Get(updater.Job.Spec.Trainer.ReplicaSpec.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			log.Warning("sync scale trainingJob to update trainingJob status error namespace=%v name=%v error=%v ",
				updater.Job.Namespace, updater.Job.Name, err.Error())
			return
		} else if err != nil {
			log.Warning("sync scale trainingJob to update trainingJob status error namespace=%v name=%v error=%v ",
				updater.Job.Namespace, updater.Job.Name, err.Error())
			continue
		}
		if (j.Status.Active + j.Status.Succeeded) >= *updater.Job.Spec.Trainer.ReplicaSpec.Spec.Parallelism {
			if updater.Status.Phase == padv1.TrainingJobPhaseSucceeded || updater.Status.Phase == padv1.TrainingJobPhaseFailed {
				return
			}
			updater.Status.Phase = padv1.TrainingJobPhaseRunning
			err := updater.updateCRDStatus()
			if err != nil {
				log.Warning("sync scale trainingJob to update trainingJob status error namespace=%v name=%v error=%v ",
					updater.Job.Namespace, updater.Job.Name, err.Error())
			}
			return
		}
	}
}

// scaleTrainingJob scale job up or down
func (updater *TrainingJobUpdater) scaleTrainingJob(additional int32) {

	// The scale action will be omit if job have done.
	if updater.Status.Phase == padv1.TrainingJobPhaseSucceeded || updater.Status.Phase == padv1.TrainingJobPhaseFailed {
		log.Infof("Omit scale for job have done, namespace=%v name=%v additional=%v ", updater.Job.Namespace, updater.Job.Name, additional)
		return
	}

	// Scale job
	scaleRecord := updater.scale(additional)
	err := updater.updateCRDStatus()
	if err != nil {
		log.Warning("scale trainingJob to update trainingJob status error namespace=%v name=%v error=%v ", updater.Job.Namespace, updater.Job.Name, err.Error())
	}

	if scaleRecord.Status == padv1.ScaleTrue {
		// Sync scale
		go updater.syncScale()
	}
}

// Start is the main process of life cycle of a TrainingJob, including create resources, event process handle and
// status convert.
func (updater *TrainingJobUpdater) start() {
	log.Infof("start updater, namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
	go updater.InitResource()

	ticker := time.NewTicker(convertedTimerTicker)
	defer ticker.Stop()
	log.Infof("start ticker, namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
	for {
		select {
		case ev := <-updater.EventCh:
			switch ev.pet {
			case trainingJobEventDelete:
				log.Infof("Delete updater, namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
				if err := updater.deleteTrainingJob(); err != nil {
					log.Errorf(err.Error())
				}
				return
			case trainingJobEventScale:
				log.Infof("Scale job, namespace=%v name=%v additional=%v: ", updater.Job.Namespace, updater.Job.Name, updater.Additional)
				updater.scaleTrainingJob(updater.Additional)
			}
		case <-ticker.C:
			updater.Convert()
			if updater.Status.Phase == padv1.TrainingJobPhaseSucceeded || updater.Status.Phase == padv1.TrainingJobPhaseFailed {
				if ticker != nil {
					log.Infof("stop ticker for job has done, namespace=%v name=%v: ", updater.Job.Namespace, updater.Job.Name)
					ticker.Stop()
				}
			}
		}
	}
}
