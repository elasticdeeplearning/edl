package master

import (
	"context"
	"google.golang.org/grpc"
	"log"
	"net"
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
	Task Task
	// A task fails if it's timeout or trainer reports it exits unnormally.
	NumFailure int
}

type masterState struct {
	Todo     []taskEntry
	Pending  map[int]taskEntry // map from task ID to task entry
	Done     []taskEntry
	Failed   []taskEntry
	CurEpoch int
}

type launcher struct {
	PodID    string
	Endpoint string
}

// Service is the master server service.
type Service struct {
	UnimplementedMasterServer

	timeoutDur time.Duration
	failureMax int

	ready    chan struct{}
	initDone bool

	mu sync.Mutex

	// store Store
	state masterState

	Chunks map[string][]Chunk // DataServerID->ChunksArray

	dataServers []DataServer
	trainers    []Trainer
	launchers   []Launcher

	etcd EtcdClient
}

// NewService creates a new service.
func NewService(etcd EtcdClient, timeoutDur time.Duration, failureMax int) (*Service, error) {
	s := &Service{}
	s.timeoutDur = timeoutDur
	s.failureMax = failureMax
	s.state.Pending = make(map[int]taskEntry)
	s.ready = make(chan struct{})
	s.etcd = etcd
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

// GetSubDataSet implements the proto interface.
func (s *Service) GetSubDataSet(context.Context, *SubDataSetRequest) (*SubDataSetResponse, error) {
	// return file from file list data set
	// Or the data can't be accessed
	return nil, nil
}

// ReportChunks implementes the proto interface.
func (s *Service) ReportChunks(ctx context.Context, in *Chunks) (*RPCRet, error) {
	return nil, nil
}

func partition(chunks []Chunk, chunksPerTask int) []taskEntry {
	return nil
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

func readChunks(globPaths []string) ([]Chunk, error) {
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
		"curPass":    s.state.CurPass,
	}
}

// GetTask gets a new task from the service.
// passID is the client side pass count
func (s *Service) GetTask(passID int, task *Task) error {
	return nil
}

// TaskFinished tell the service that a task is finished.
func (s *Service) TaskFinished(taskID int, dummy *int) error {
	return nil
}

// TaskFailed tells the service that a task is failed.
func (s *Service) TaskFailed(meta TaskMeta, dummy *int) error {
	return nil
}
