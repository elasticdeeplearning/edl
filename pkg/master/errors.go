package master

import (
	"fmt"

	pb "github.com/paddlepaddle/edl/pkg/masterpb"
)

// ErrorType is the typei name of error.
type ErrorType string

// Error implements Error interface
type Error struct {
	Type   ErrorType
	Detail string
}

const (
	// DuplicateInitDataSetError is used to reported dataset error.
	DuplicateInitDataSetError ErrorType = "DuplicateInitDataSetError"
	// BarrierError is used to barrier
	BarrierError ErrorType = "BarrierError"
)

// String converts a ErrorType into its corresponding canonical error message.
func (t ErrorType) String() string {
	switch t {
	case DuplicateInitDataSetError:
		return "DataSet must be inited once."
	case BarrierError:
		return "Can't barrier now"
	default:
		panic(fmt.Sprintf("unrecognized validation error: %q", string(t)))
	}
}

// Error implements the error interface.
func (v *Error) Error() string {
	return fmt.Sprintf("%s", v.errorBody())
}

func (v *Error) errorBody() string {
	s := v.Type.String()
	if len(v.Detail) != 0 {
		s += fmt.Sprintf(": %s", v.Detail)
	}
	return s
}

// ToRPCRet converts Error to RPCRet
func (v *Error) ToRPCRet() *pb.RPCRet {
	ret := &pb.RPCRet{}
	ret.Type = string(v.Type)
	ret.Detail = v.Detail
	return ret
}

// DuplicateInitDataSet make the correspond error.
func DuplicateInitDataSet(detail string) *Error {
	return &Error{DuplicateInitDataSetError, detail}
}
