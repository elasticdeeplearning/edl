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

// Service is the master server service.
type Service struct {
	UnimplementedMasterServer
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
