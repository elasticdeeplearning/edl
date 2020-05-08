package master

import (
	"fmt"
)

// ErrorType is the typei name of error.
type ErrorType string

// Error implements Error interface
type Error struct {
	Type   ErrorType
	Detail string
}

const (
	// ErrorTypeDuplicateInitDataSet is used to reported dataset error.
	ErrorTypeDuplicateInitDataSet ErrorType = "DuplicateInitDataSet"
)

// String converts a ErrorType into its corresponding canonical error message.
func (t ErrorType) String() string {
	switch t {
	case ErrorTypeDuplicateInitDataSet:
		return "DataSet must be inited once."
	default:
		panic(fmt.Sprintf("unrecognized validation error: %q", string(t)))
	}
}

// Error implements the error interface.
func (v *Error) Error() string {
	return fmt.Sprintf("%s", v.ErrorBody())
}

func (v *Error) errorBody() string {
	s := v.Type.String()
	if len(v.Detail) != 0 {
		s += fmt.Sprintf(": %s", v.Detail)
	}
	return s
}

// DuplicateInitDataSet make the correspond error.
func DuplicateInitDataSet(detail string) *Error {
	return &Error{ErrorTypeDuplicateInitDataSet, detail}
}
