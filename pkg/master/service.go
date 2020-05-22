package master

import (
	"context"
	"fmt"
	log "github.com/inconshreveable/log15"
	pb "github.com/paddlepaddle/edl/pkg/masterpb"
	"sync"
	"time"
)

const (
	port = ":50051"
)

// Store is the interface for save and load the master state.
type Store interface {
	Save([]byte) error
	Load() ([]byte, error)
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

// Service is the master server service.
type Service struct {
	// pb.UnimplementedMasterServer

	timeoutDur time.Duration
	failureMax int

	ready    chan struct{}
	initDone bool

	mu sync.Mutex

	// store Store
	state masterState

	Chunks map[string][]pb.Chunk // DataServerID->ChunksArray

	etcd EtcdClient

	dataset map[string]pb.DataSet
}

// NewService creates a new service.
func NewService(etcd *EtcdClient, timeoutDur time.Duration, failureMax int) (*Service, error) {
	s := &Service{}
	s.timeoutDur = timeoutDur
	s.failureMax = failureMax
	s.state.Pending = make(map[int]taskEntry)
	s.ready = make(chan struct{})
	s.etcd = *etcd
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
// TODO
func (s *Service) recover() (bool, error) {
	return true, nil
}

// snapshot *must* be called with s.mu being held.
// TODO
func (s *Service) snapshot() error {
	return nil
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
