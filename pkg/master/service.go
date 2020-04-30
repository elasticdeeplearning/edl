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

// TaskMeta is a struct which stores task's meta info.
type TaskMeta struct {
	ID    int
	Epoch int
	Stage string
}

// Task is the basic unit of data instances assigned to trainers.
type Task struct {
	Meta   TaskMeta
	Chunks []Chunk
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

// Service is the master server service.
type Service struct {
	UnimplementedMasterServer

	timeoutDur time.Duration
	failureMax int

	ready    chan struct{}
	initDone bool

	mu sync.Mutex
}

// Chunk is a chunk of data consisted of several data instances.
type Chunk struct {
	DataServer    string
	IdxInFileList int64
	FilePath      string
	RecordNo      []int64
}

// NewService creates a new service.
func NewService(store Store, timeoutDur time.Duration, failureMax int) (*Service, error) {
	s := &Service{}
	s.timeoutDur = timeoutDur
	s.failureMax = failureMax
	s.state.Pending = make(map[int]taskEntry)
	s.ready = make(chan struct{})
	s.store = store
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

	return s, nil
}

// GetSubDataSet implements the proto interface.
func (*Service) GetSubDataSet(context.Context, *SubDataSetRequest) (*SubDataSet, error) {
	return nil, nil
}

func main() {
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	RegisterMasterServer(s, &Service{})
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
