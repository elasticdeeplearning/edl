package autoscaler

import (
	"k8s.io/api/core/v1"
)

// AddResourceList add another v1.ResourceList to first's inner
// quantity.  v1.ResourceList is equal to map[string]Quantity
func AddResourceList(a v1.ResourceList, b v1.ResourceList) {
	for resname, q := range b {
		v, ok := a[resname]
		if !ok {
			a[resname] = q.DeepCopy()
		}
		v.Add(q)
		a[resname] = v
	}

	return
}
