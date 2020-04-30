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

type fileDataSet struct {
	IdxInFileList int64
	FilePath      string
}

// DataSet represents a file list dataset.
type DataSet struct {
	Files []fileDataSet
}

// NewDataSet constructs a new DataSet.
func NewDataSet(fileList []string) (*DataSet, error) {
	d := &DataSet{}

	for i, v := range fileList {
		o := fileDataSet{}

		o.IdxInFileList = i
		o.FilePath = v

		d.Files = append(d.Files, o)
	}

	return d, nil
}

// StartNewEpoch starts a new epoch
func (d *DataSet) StartNewEpoch() {
	return
}

// GetFile gets one file from the file list
func (d *DataSet) GetFile() (string, error) {
	return nil, nil
}
