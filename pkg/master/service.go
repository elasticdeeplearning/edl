package master

import (
	"context"
	"fmt"
	"sync"
	"time"

	"bytes"
	"compress/gzip"
	"encoding/gob"

	log "github.com/inconshreveable/log15"
	pb "github.com/paddlepaddle/edl/pkg/masterpb"
)

// Store is the interface for save and load the master state.
type Store interface {
	Save([]byte, string) error
	Load(string) ([]byte, error)
	Shutdown() error
}

type taskEntry struct {
	Task pb.Task
	// A task fails if it's timeout or trainer reports it exits unnormally.
	NumFailure int
}

type masterState struct {
	Todo     []taskEntry
	Pending  map[int]taskEntry // map from task ID to task entry
	Done     []taskEntry
	Failed   []taskEntry
	CurEpoch int
	Stage    string // generate by master and Pods change job stage change.
	version  string // should equal with the trainer's checkpoint
}

type launcher struct {
	PodID    string
	Endpoint string
}

type masterMeta struct {
	Endpoint string
	JobStage string
	OldStage map[string][]string
}

// Service is the master server service.
type Service struct {
	// pb.UnimplementedMasterServer
	jobID string

	timeoutDur time.Duration
	failureMax int

	ready    chan struct{}
	initDone bool

	mu sync.Mutex

	store Store
	state masterState
	meta  masterMeta

	Chunks map[string][]pb.Chunk // DataServerID->ChunksArray

	etcd EtcdClient

	dataset map[string]pb.DataSet
}

// NewService creates a new service.
func NewService(jobID string, etcd *EtcdClient, timeoutDur time.Duration, failureMax int) (*Service, error) {
	s := &Service{}
	s.timeoutDur = timeoutDur
	s.failureMax = failureMax
	s.state.Pending = make(map[int]taskEntry)
	s.ready = make(chan struct{})
	s.etcd = *etcd
	s.jobID = jobID
	if etcd != nil {
		recovered, err := s.recover()
		if err != nil {
			return nil, err
		}

		if recovered {
			// Recovered. Now the state is already initialized,
			// and the master is ready.
			s.initDone = true
			close(s.ready)
			log.Info("Master recovered from saved state.")
		}
	}

	return s, nil
}

func (s *Service) watchCluster() {
}

// Register record this endpoint to etcd so others can find it.
func (s *Service) Register(endpoint string) error {
	return nil
}

// GetSubDataSet implements the proto interface.
func (s *Service) GetSubDataSet(context.Context, *pb.SubDataSetRequest) (*pb.SubDataSetResponse, error) {
	// return file from file list data set
	// Or the data can't be accessed
	return nil, nil
}

// ReportChunks implementes the proto interface.
func (s *Service) ReportChunks(ctx context.Context, in *pb.DataServerChunk) (*pb.RPCRet, error) {
	return nil, nil
}

// Barrier implementes the proto interface.
func (s *Service) Barrier(ctx context.Context, in *pb.BarrierRequest) (*pb.ClusterResponse, error) {
	return nil, nil
}

// recover recovers service state from etcd.
func (s *Service) recoverState() (bool, error) {
	state, err := s.store.Load(statePath(s.jobID))
	if err != nil {
		return false, err
	}

	if state == nil {
		log.Info("No state exists, not recovered.")
		return false, nil
	}

	log.Info("Loaded snapshot.", log.Ctx{"size": len(state)})
	gr, err := gzip.NewReader(bytes.NewReader(state))
	if err != nil {
		return false, err
	}

	dec := gob.NewDecoder(gr)
	var tqs masterState
	err = dec.Decode(&tqs)
	if err != nil {
		return false, err
	}

	err = gr.Close()
	if err != nil {
		// Only close failed, recover actually succeed, so
		// just log error.
		log.Error("error close recover file.", log.Ctx{"error": err})
	}

	s.state = tqs
	log.Info("Master recovered from snapshot, scheduling pending task timeout check.", s.logCtx())
	for _, t := range s.state.Pending {
		time.AfterFunc(s.timeoutDur, s.checkTimeoutFunc(t.Task.Meta.ID, t.Task.Meta.Epoch))
	}

	return true, nil
}

func (s *Service) recoverMeta() (bool, error) {
	d, err := s.store.Load(metaPath(s.jobID))
	if err != nil {
		return false, err
	}

	if d == nil {
		log.Info("No state exists, not recovered.")
		return false, nil
	}

	dec := gob.NewDecoder(r)
	var m masterMeta
	err = dec.Decode(&m)
	if err != nil {
		message = ""
		return false, err
	}

	log.Info("Loaded master meta.", log.Ctx{"size": len(m)})
	return true, nil
}

func (s *Service) recover() (bool, error) {
	r, err := s.recoverState()
	if err != nil {
		return r, err
	}

	r, err = s.recoverMeta()
	if err != nil {
		return r, err
	}

	return true, nil
}

// snapshot *must* be called with s.mu being held.
func (s *Service) snapshot(v interface{}, path string) error {
	// TODO(helin): etcd request has a size limit, so the snapshot
	// size is limited by the max request size. We should either
	// divide the snapshot into smaller chunks and save under
	// different keys, or configure the request size to be big
	// enough:
	// https://github.com/coreos/etcd/blob/2f84f3d8d8ed8f9537ab6ffa44a3a1c7eddfa9b1/embed/config.go#L44
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	enc := gob.NewEncoder(gw)
	err := enc.Encode(v)
	if err != nil {
		return err
	}
	err = gw.Close()
	if err != nil {
		return err
	}

	state := buf.Bytes()
	log.Info("Saving snapshot.", log.Ctx{"size bytes": len(state)})
	return s.store.Save(s.jobId, state, path)
}

func readChunks(globPaths []string) ([]pb.Chunk, error) {
	return nil, nil
}

// SetDataSet implements the proto interface.
func (s *Service) SetDataSet(globPaths []string, _ *int) error {
	return nil
}

// processFailedTask retry s.failureMax times for failed task.
// return true if all task are done or failed.
func (s *Service) processFailedTask(t taskEntry, epoch int) {
	return
}

// processFailedTask retry s.failureMax times for failed task.
// return true if all task are done or failed.
func (s *Service) GetID(ctx context.Context, in *pb.EmptyRequest) (*pb.Entity, error) {
	return nil, nil
}

func (s *Service) checkTimeoutFunc(taskID int, epoch int) func() {
	return func() {
		s.mu.Lock()
		defer s.mu.Unlock()

		t, ok := s.state.Pending[taskID]
		if !ok {
			return
		}

		s.processFailedTask(t, epoch)
	}
}

// must be called with lock held.
func (s *Service) logCtx() log.Ctx {
	return log.Ctx{
		"todoLen":    len(s.state.Todo),
		"pendingLen": len(s.state.Pending),
		"doneLen":    len(s.state.Done),
		"failedLen":  len(s.state.Failed),
		"curPass":    s.state.CurEpoch,
	}
}

// GetTask gets a new task from the service.
// passID is the client side pass count
func (s *Service) GetTask(context.Context, *pb.TaskRequest) (*pb.TaskResponse, error) {
	return nil, nil
}

// TaskFailed tells the service that a task is failed.
func (s *Service) TaskFailed(meta pb.TaskMeta, dummy *int) error {
	return nil
}

// AddDataSet adds a initial dataset to service.
func (s *Service) AddDataSet(ctx context.Context, in *pb.DataSet) (*pb.RPCRet, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dataset != nil {
		if _, ok := s.dataset[in.Name]; ok {
			return DuplicateInitDataSet(fmt.Sprintf("dataset:%v", in.Name)).ToRPCRet(), nil
		}
	} else {
		s.dataset = make(map[string]pb.DataSet)
		s.dataset[in.Name] = *in
	}

	return &pb.RPCRet{}, nil
}

// GetCluster gets cluster elements from the service.
func (s *Service) GetCluster(ctx context.Context, in *pb.ClusterRequest) (*pb.ClusterResponse, error) {
	return nil, nil
}

// NewEpoch starts a new epoch of the service.
func (s *Service) NewEpoch(ctx context.Context, in *pb.NewEpochRequest) (*pb.RPCRet, error) {
	return nil, nil
}

// TaskErrored reports a new tasks error to the service.
func (s *Service) TaskErrored(ctx context.Context, in *pb.Tasks) (*pb.RPCRet, error) {
	return nil, nil
}

// TaskFinished tell the service that a task is finished.
func (s *Service) TaskFinished(ctx context.Context, in *pb.Tasks) (*pb.RPCRet, error) {
	return nil, nil
}
